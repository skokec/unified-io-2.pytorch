import os
import torch

from torch.nn import functional as F
from uio2.model import UnifiedIOModel
from uio2.preprocessing import build_batch, UnifiedIOPreprocessor

from tqdm import tqdm

from demo import centers_to_tokens

from datasets import ClothDataset
from utils.utils import variable_len_collate

def train_step(model, preprocessor, sample):

    preprocessed_examples = []
    for i in range(len(sample['im_name'])):
        im_size = [ii[i] for ii in sample['im_size']]
        center = sample['center'][i]

        gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]

        gt_centers_text = centers_to_tokens(gt_centers, (im_size[1],im_size[0]))

        imput = "List coordinates of all visible towel corners in <image_input>"
        output_target = gt_centers_text

        preprocessed_example = preprocessor(text_inputs=imput, image_inputs=sample['im_name'][i], text_targets=output_target, target_modality="text")
        preprocessed_examples.append(preprocessed_example)

    batch = build_batch(preprocessed_examples, device=model.device)
    out = model(batch)
    total_loss = 0

    for modality, (logits, targets, mask) in out.items():
        losses = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1).to(torch.long), reduction="none")        
        total_loss += (losses.reshape(logits.shape[:2])*mask)/mask.sum()
    
    return total_loss


def train():

    LLAMA2_TOKENIZER = os.environ['LLAMA2_TOKENIZER']
    VICOS_TOWEL_DATASET = os.environ['VICOS_TOWEL_DATASET']

    dataset_workers = 4
    batch_size = 2

    learning_rate = 0.001
    weight_decay = 0.00001

    #############################################################################################
    # DATASET
    from torchvision.transforms import InterpolationMode
    transform_img_keys = ['image', 'instance', 'label', 'ignore', 'orientation', 'mask']
    transform_tensors = [torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor]
    transform_img_interpolate = [InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST,  InterpolationMode.NEAREST,  InterpolationMode.BILINEAR, InterpolationMode.NEAREST]

    USE_SEGMENTATION = False
    USE_DEPTH = False

    if USE_SEGMENTATION:
        transform_img_keys.append('segmentation_mask')
        transform_tensors.append(torch.FloatTensor)
        transform_img_interpolate.append(InterpolationMode.NEAREST)

    if USE_DEPTH:
        transform_img_keys.append('depth')
        transform_tensors.append(torch.FloatTensor)
        transform_img_interpolate.append(InterpolationMode.NEAREST)

    transform = [
        # for training without augmentation (same as testing)
        {
            'name': 'ToTensor',
            'opts': {
                'keys': transform_img_keys,
                'type': transform_tensors,
            },			
        },
    ]    
    subfolders = [dict(folder='bg=white_desk', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
	   dict(folder='bg=green_checkered', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
	   dict(folder='bg=poster', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
	   dict(folder='bg=festive_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
    ]

	#subfolders = [dict(folder='../demo_cube', annotation='annotations.json', data_subfolders=['.']),]
	#subfolders = [dict(folder='../IJS-examples',data_subfolders=['.'])]

    train_dataset = ClothDataset(root_dir=VICOS_TOWEL_DATASET, 
                                 resize_factor=1, transform_only_valid_centers=1.0, transform=transform, use_depth=USE_DEPTH, segment_cloth=USE_SEGMENTATION, correct_depth_rotation=False, subfolders=subfolders)

    train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=dataset_workers, pin_memory=True,
                                                   collate_fn=variable_len_collate)

    #############################################################################################
    # MODEL
    dev0 = torch.device("cuda:0")

    preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor", tokenizer=LLAMA2_TOKENIZER)

    #model = UnifiedIOModel.from_pretrained("allenai/uio2-xxl")
    model = UnifiedIOModel.from_pretrained("allenai/uio2-large")
    #model = UnifiedIOModel.from_pretrained("allenai/uio2-large-bfloat16")
    model.to(dev0)
    model.set_dev1(dev0)
    model.set_dev2(dev0)

    #############################################################################################
    # OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)

    model.train()

    NUM_EPOCH = 1
    for epoch in range(NUM_EPOCH):

        tqdm_iterator = tqdm(train_dataset_it, desc="Training epoch #%d/%d" % (epoch,NUM_EPOCH),dynamic_ncols=True)

        for sample in tqdm_iterator:
            #with torch.no_grad():
            loss = train_step(model, preprocessor, sample)
            loss = loss.sum()

            if False:
                per_devices_params = {dev0:0, dev1:0}
                #print("Devices for all parameters in the model:")        
                for name, param in model.named_parameters():
                    #print(f"Parameter: {name}, Device: {param.device}")
                    per_devices_params[param.device] += torch.prod(torch.tensor(param.shape)).item()
                tqdm_iterator.set_postfix(**{f"num param({k})":f"{v/(1024*1024)*4} MB" for k,v in per_devices_params.items()})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            
            tqdm_iterator.set_postfix(loss=loss.item())
  


if __name__ == "__main__":    
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train()