import datetime
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
from matplotlib import pyplot as plt
from tqdm import tqdm

from torch.nn import functional as F

from uio2.model import UnifiedIOModel
from uio2.preprocessing import build_batch, UnifiedIOPreprocessor
from utils.utils import distributed_sync_dict, variable_len_collate, AverageMeter, Logger

from demo import centers_to_tokens

from config import get_config_args
from datasets import get_dataset

class Trainer:
    def __init__(self, local_rank, rank_offset, world_size, args, use_distributed_data_parallel=True, attach_debug=False):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        self.args = args
        self.world_size = world_size
        self.world_rank = rank_offset + local_rank
        self.local_rank = local_rank

        self.use_distributed_data_parallel = use_distributed_data_parallel and world_size > 1

        self.attach_debug = attach_debug

        if args['save'] and self.world_rank == 0:
            if not os.path.exists(args['save_dir']):
                os.makedirs(args['save_dir'])

            # save parameters
            with open(os.path.join(args['save_dir'],'params.json'), 'w') as file:
                file.write(json.dumps(args, indent=4, sort_keys=True,default=lambda o: '<not serializable>'))

        if args.get('display'):
            plt.ion()
        else:
            plt.ioff()
            plt.switch_backend("agg")

    def initialize_data_parallel(self, init_method=None):
        ###################################################################################################
        # set device
        if self.use_distributed_data_parallel and init_method is not None:
            self.use_distributed_data_parallel = True

            self.device = torch.device("cuda:%d" % self.local_rank)

            # if not master, then wait at least 5 sec to give master a chance for starting up first
            if self.world_rank != 0:
                time.sleep(10+np.random.randint(0,10))
            elif self.attach_debug:
                import ptvsd

                # Allow other computers to attach to the debugger
                ptvsd.enable_attach(address=('0.0.0.0', self.attach_debug))
                print(f"Waiting for debugger attach at port {self.attach_debug}...")
                ptvsd.wait_for_attach()
            # initialize the process group
            torch.distributed.init_process_group("nccl", init_method=init_method, timeout=datetime.timedelta(hours=1),
                                                 rank=self.world_rank, world_size=self.world_size)

            print('Waiting for all nodes (ready from rank=%d/%d)' % (self.world_rank, self.world_size))
            sys.stdout.flush()
            torch.distributed.barrier(device_ids=[self.local_rank])
        else:
            if self.attach_debug:
                import ptvsd

                # Allow other computers to attach to the debugger
                ptvsd.enable_attach(address=('0.0.0.0', self.attach_debug))
                print(f"Waiting for debugger attach at port {self.attach_debug}...")
                ptvsd.wait_for_attach()

            self.use_distributed_data_parallel = False
            self.device = torch.device("cuda" if self.args['cuda'] else "cpu")

        #torch.backends.cudnn.benchmark = True

    def cleanup(self):
        if self.use_distributed_data_parallel:
            torch.distributed.destroy_process_group()

    def _to_data_parallel(self, X, **kwargs):
        if self.use_distributed_data_parallel:
            X = torch.nn.parallel.DistributedDataParallel(X.to(self.device), device_ids=[self.local_rank], find_unused_parameters=True, gradient_as_bucket_view=True, **kwargs)
        else:
            X = torch.nn.DataParallel(X.to(self.device), device_ids=[i for i in range(self.world_size)], **kwargs)
        return X

    def _synchronize_dict(self, array):
        if self.use_distributed_data_parallel:
            array = distributed_sync_dict(array, self.world_size, self.world_rank, self.device)
        return array

    def initialize(self):
        args = self.args
        device = self.device

        ###################################################################################################
        # train dataloader
        dataset_workers = args['train_dataset']['workers'] if 'workers' in args['train_dataset'] else 0
        dataset_batch = args['train_dataset']['batch_size'] if 'batch_size' in args['train_dataset'] else 1
        dataset_shuffle = args['train_dataset']['shuffle'] if 'shuffle' in args['train_dataset'] else True

        self.accumulate_grads_iter = args['model'].get('accumulate_grads_iter',1)
        if self.accumulate_grads_iter:
            dataset_batch = dataset_batch // self.accumulate_grads_iter


        # in distributed settings we need to manually reduce batch size
        if self.use_distributed_data_parallel:
            dataset_batch = dataset_batch // self.world_size
            if not args['train_dataset'].get('force_workers_on_distributed_processing'):
                dataset_workers = 0 # ignore workers request since already using separate processes for each GPU

        #############################################################################################
        # DATASET
        preprocessor = UnifiedIOPreprocessor.from_pretrained(args['model']['preprocessor'], **args['model']['preprocessor_kwargs'])

        #train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'], preprocessor=preprocessor)
        train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'], preprocessor=None)

        train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_size=dataset_batch, num_workers=dataset_workers, 
                                                       pin_memory=True if args['cuda'] else False, shuffle=dataset_shuffle, 
                                                       collate_fn=variable_len_collate)

        ###################################################################################################
        # set model
        
        model = UnifiedIOModel.from_pretrained(args['model']['name'])
        model.to(device)
        model.set_dev1(device)
        model.set_dev2(device)

        model = self._to_data_parallel(model, dim=0)

        def get_optimizer(model_, args_):
            if args_ is None or args_.get('disabled'):
                return None, None
            if 'optimizer' not in args_ or args_['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(model_.parameters(),lr=args_['lr'],
                                             weight_decay=args_['weight_decay'])
            elif args_['optimizer'] == 'SGD':
                optimizer = torch.optim.SGD(model_.parameters(),lr=args_['lr'],
                                            momentum=args_['momentum'],
                                            weight_decay=args_['weight_decay'])
            # use custom lambda_scheduler_fn function that can pass args if available
            lr_lambda = args_['lambda_scheduler_fn'](args) if 'lambda_scheduler_fn' in args_ else args_['lambda_scheduler']
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            return optimizer, scheduler

        # set optimizer for model and for center model
        optimizer, scheduler = get_optimizer(model, args['model'])

        if args.get('resume') and not args.get('resume_path'):
            args['resume_path'] = self._get_last_checkpoint()

        ########################################################################################################################
        # Logger
        self.logger = Logger(('train', ), 'loss')

        # resume
        self.start_epoch = 0
        if args['resume_path'] is not None and os.path.exists(args['resume_path']):
            print('Resuming model from {}'.format(args['resume_path']))
            state = torch.load(args['resume_path'])
            self.start_epoch = state['epoch'] + 1
            self.logger.data = state['logger_data']
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if 'optim_state_dict' in state and optimizer: optimizer.load_state_dict(state['optim_state_dict'])

        if args.get('pretrained_model_path') is not None and os.path.exists(args['pretrained_model_path']):
            print('Loading pre-trained model from {}'.format(args['pretrained_model_path']))
            state = torch.load(args['pretrained_model_path'])
            if 'model_state_dict' in state:
                missing, unexpected = model.load_state_dict(state['model_state_dict'], strict=False)
                if len(missing) > 0 or len(unexpected) > 0:
                    print('WARNING: #####################################################################################################')
                    print('WARNING: Current model differs from the pretrained one, loading weights using strict=False')
                    print('WARNING: #####################################################################################################')

        self.device = device
        self.train_dataset_it, self.dataset_batch = train_dataset_it, dataset_batch
        self.model, self.preprocessor = model, preprocessor
        self.optimizer, self.scheduler = optimizer, scheduler

    def do_tf_logging(self, iter, type):
        if self.world_rank != 0:
            return False

        args = self.args
        if 'tf_logging' not in args:
            return False

        if 'tf_logging' in args and type not in args['tf_logging']:
            return False

        if 'tf_logging_iter' in args and iter % args['tf_logging_iter'] != 0:
            return False

        return True

    def print(self, *kargs, **kwargs):
        if self.world_rank == 0:
            print(*kargs, **kwargs)

    def train(self, epoch):
        args = self.args

        device = self.device
        train_dataset_it, dataset_batch = self.train_dataset_it, self.dataset_batch
        model, preprocessor = self.model, self.preprocessor
        optimizer = self.optimizer

        # sum over channels and spatial location
        reduction_dim = (1,2,3)

        # put model into training mode
        model.train()
       
        # define meters
        loss_meter = AverageMeter()

        if optimizer:
            for param_group in optimizer.param_groups:
                self.print('learning rate (model): {}'.format(param_group['lr']))


        iter=epoch*len(train_dataset_it)

        all_samples_metrics = {}
        tqdm_iterator = tqdm(train_dataset_it, desc="Training epoch #%d/%d" % (epoch,args['n_epochs']),dynamic_ncols=True) if self.world_rank == 0 else None

        train_preprocessor_args = args['train_dataset'].get('keypoint_preprocesser_kwargs')
        if not train_preprocessor_args:
            train_preprocessor_args = dict()

        from datasets.PreprocessorDataset import KeypointPreprocessorDataset
        train_preprocessor = KeypointPreprocessorDataset(preprocessor=preprocessor, dataset=None, **train_preprocessor_args)


        for i, sample in enumerate(tqdm_iterator if tqdm_iterator is not None else train_dataset_it):
            
            # run train_preprocessor inside this loop to use GPU instead of running it as dataset
            if '/inputs/image/input' not in sample:
                preprocessed_examples = []
                for i in range(len(sample['im_name'])):
                    single_sample = dict(center=sample['center'][i],
                                         image=sample['image'][i])
                    
                    preprocessed_example = train_preprocessor.preprocess_sample(single_sample)
                    
                    preprocessed_examples.append(preprocessed_example)
                
                batch = build_batch(preprocessed_examples, device=device)
            else:
                batch = sample
            
            out = model(batch)
            
            total_loss = 0
            for modality, (logits, targets, mask) in out.items():
                out_res = torch.argmax(torch.softmax(logits,dim=2),dim=2)
                diff_tokens = out_res - targets
                losses = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1).to(torch.long), reduction="none")        
                total_loss += (losses.reshape(logits.shape[:2])*mask)/mask.sum()

            # since each GPU will have only portion of data it will not use correct batch size for averaging - do correction for this here
            if self.world_size > 1 and not self.use_distributed_data_parallel:
                total_loss = [l/float(self.world_size) for l in total_loss]

            # save losses and metrics from this batch to common storage for this epoch
            all_samples_metrics = self._updated_per_epoch_sample_metrics(all_samples_metrics, sample['index'],
                                                                         total_loss, metrics=None)
            loss = total_loss.sum()

            # we can simply sum the final loss since average is already calculated through weighting
            bp_loss = loss / self.accumulate_grads_iter
            bp_loss.backward()

            if self.do_tf_logging(iter, 'weights'):
                self.visualizer.log_conv_weights(model, iter=iter)

            if ((i + 1) % self.accumulate_grads_iter == 0) or (i + 1 == len(train_dataset_it)):
                if optimizer:
                    optimizer.step()
                    optimizer.zero_grad() # set_to_none=False for prior to v2.0 and set_to_none=True after v2.0

            loss_meter.update(loss.item())

            if tqdm_iterator is not None:
                tqdm_iterator.set_postfix(loss=loss.item())

            all_samples_metrics = self._synchronize_dict(all_samples_metrics)

            iter+=1

        all_samples_total_loss = {k:v['loss'] for k,v in all_samples_metrics.items()}

        if epoch == args['n_epochs']:
            self.print('end')
        
        return np.array(list(all_samples_total_loss.values())).mean() * dataset_batch

    def _updated_per_epoch_sample_metrics(self, stored_results, sample_indexes, total_loss, metrics=None):

        for b in range(len(total_loss)):
            index = sample_indexes[b].item()

            # add losses
            stored_results[index] = dict(loss=total_loss[b].sum().item())

            if metrics is not None:
                stored_results[index].update(metrics[b])

        return stored_results


    def save_checkpoint(self, state, is_best=False, name='checkpoint.pth'):
        args = self.args

        print('=> saving checkpoint')
        file_name = os.path.join(args['save_dir'], name)
        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name, os.path.join(args['save_dir'], 'best_iou_model.pth'))
        if state['epoch'] % args.get('save_interval',10) == 0:
            shutil.copyfile(file_name, os.path.join(args['save_dir'], 'checkpoint_%03d.pth' % state['epoch']))

    def should_skip_training(self):
        last_interval = self.args['n_epochs'] - self.args.get('save_interval',10)
        last_checkpoint = os.path.join(self.args['save_dir'], 'checkpoint_%03d.pth' % last_interval)

        return self.args.get('skip_if_exists') and os.path.exists(last_checkpoint)

    def _get_last_checkpoint(self):
        valid_last_checkpoint = None

        recent_checkpoints = [os.path.join(self.args['save_dir'], 'checkpoint.pth')]
        recent_checkpoints += [os.path.join(self.args['save_dir'], 'checkpoint_%03d.pth' % epoch) for epoch in list(range(self.args['n_epochs']))[::-1]]

        for last_checkpoint in recent_checkpoints:
            if os.path.exists(last_checkpoint):
                valid_last_checkpoint = last_checkpoint
                break
        return valid_last_checkpoint

    def run(self):
        args = self.args

        for epoch in range(self.start_epoch, args['n_epochs']):

            train_loss = self.train(epoch)

            if self.world_rank == 0: print('Starting epoch {}'.format(epoch))
            if self.scheduler: self.scheduler.step()

            if self.world_rank == 0:
                print('===> train loss: {:.2f}'.format(train_loss))

                self.logger.add('train', train_loss)
                self.logger.plot(save=args['save'], save_dir=args['save_dir'])

                if args['save'] and (epoch % args.get('save_interval',10) == 0 or epoch + 1 == args['n_epochs']):
                    state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict() if self.model is not None else None,
                        'optim_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                        'logger_data': self.logger.data
                    }
                    self.save_checkpoint(state)

def main(local_rank, rank_offset, world_size, init_method=None):

    args = get_config_args()

    trainer = Trainer(local_rank, rank_offset, world_size, args, use_distributed_data_parallel=init_method is not None, attach_debug=args.get('attach_debug'))

    if trainer.should_skip_training():
        print('Skipping due to already existing checkpoints (and requested to skip if exists) !!')
        return

    trainer.initialize_data_parallel(init_method)

    trainer.initialize()
    trainer.run()

    trainer.cleanup()

import torch.multiprocessing as mp

if __name__ == "__main__":
    args = get_config_args()

    n_gpus = torch.cuda.device_count()
   
    world_size = int(os.environ.get('WORLD_SIZE',default=n_gpus))
    rank_offset = int(os.environ.get('RANK_OFFSET',default=0))

    if world_size <= 1 or args.get('disable_distributed_training'):
        main(0, 0, n_gpus)
    else:
        spawn = None
        try:
            print("spawning %d new processes" % n_gpus)
            spawn = mp.spawn(main,
                             args=(rank_offset,world_size,'env://'),
                             nprocs=n_gpus,
                             join=False)
            while not spawn.join():
                pass
        except KeyboardInterrupt:
            if spawn is not None:
                for pid in spawn.pids():
                    os.system("kill %s" % pid)
            torch.distributed.destroy_process_group()
