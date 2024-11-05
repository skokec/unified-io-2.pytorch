import torch

from uio2.model import UnifiedIOModel
from uio2.preprocessing import UnifiedIOPreprocessor
from uio2.runner import TaskRunner, extract_keypoints, extract_individual_keypoints
from uio2.data_utils import resize_and_pad_default, values_to_tokens
from uio2 import config

from transformers import GenerationConfig

import pylab as plt
from PIL import Image
from tqdm import tqdm
import os

from config import get_config_args

import numpy as np

def detect_keypoints(runner, img_filename, **kwargs):
    p = "List coordinates of all visible towel corners"
    example = runner.uio2_preprocessor(text_inputs=p, image_inputs=img_filename, target_modality="text")
    text = runner.predict_text(example, max_tokens=32, detokenize=True, logits_processor=None, **kwargs)
    kps = extract_individual_keypoints(text, example["/meta/image_info"])
    
    return kps, text


def centers_to_tokens(gt_centers, img_shape):
    
    gt_centers_tokens_x = values_to_tokens(gt_centers[:,0] / img_shape[1])
    gt_centers_tokens_y = values_to_tokens(gt_centers[:,1] / img_shape[0])


    gt_centers_tokens_x = [x.decode("utf-8") for x in gt_centers_tokens_x.numpy()]
    gt_centers_tokens_y = [y.decode("utf-8") for y in gt_centers_tokens_y.numpy()]

    gt_centers_text = " , ".join([f"{x} {y}" for x,y in zip(gt_centers_tokens_x, gt_centers_tokens_y)])

    return gt_centers_text


if __name__ == "__main__":

    args = get_config_args()

    PLOT = False

    #EXP_FOLDER = "./exp/vicos-towels/model=allenai/uio2-large_resize_factor=1_batchsize=24/num_train_epoch=100/depth=False"
    #EXP_FOLDER = "./exp/vicos-towels/model=allenai/uio2-large_resize_factor=1_batchsize=24_lr=0.001/num_train_epoch=100/depth=False"
    EVAL_FOLDER = args['save_dir']
    EVAL_TYPE = args['eval_type']
    EVAL_EPOCH = args['eval_epoch']
    #EVAL_TYPE = "train" # test or train
    #EVAL_EPOCH = "_100"
    

    dev = torch.device("cuda:0")

    preprocessor = UnifiedIOPreprocessor.from_pretrained(args['model']['preprocessor'], **args['model']['preprocessor_kwargs'])
    model = UnifiedIOModel.from_pretrained(args['model']['name'],local_files_only=True)

    
    state = torch.load(os.path.join(EVAL_FOLDER,f"checkpoint{EVAL_EPOCH}.pth"))

    model_state_dict = {k.replace("module.",""):v for k,v in state['model_state_dict'].items()}
    model.load_state_dict(model_state_dict, strict=True)
    model.to(dev)

    model.eval()

    runner = TaskRunner(model, preprocessor)

    if True:
        from datasets import ClothDataset, KeypointPreprocessorDataset

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
                    'keys': ('image',),
                    'type': (torch.FloatTensor,),
                },			
            },
            # {
			# 		'name': 'RandomHorizontalFlip',
			# 		'opts': {
            #             'keys': ('image',),'keys_bbox': ('center',),
			# 			#'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
			# 			#'keys_custom_fn' : { 'orientation': lambda x: (-1*x + np.pi)  % (2 * np.pi) - np.pi},
			# 			'p': 0.5,
			# 		}
			# 	},
			# 	{
			# 		'name': 'RandomVerticalFlip',
			# 		'opts': {
            #             'keys': ('image',),'keys_bbox': ('center',),
			# 			#'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
			# 			#'keys_custom_fn' : { 'orientation': lambda x: (np.pi - x + np.pi)  % (2 * np.pi) - np.pi},
			# 			'p': 0.5,
			# 		}
			# 	},
			# 	{
			# 		'name': 'RandomCustomRotation',
			# 		'opts': {
			# 			'keys': ('image',),'keys_bbox': ('center',),
			# 			'resample': (InterpolationMode.BILINEAR,),
			# 			#'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
			# 			#'keys_custom_fn' : { 'orientation': lambda x,angle: (x + (angle*np.pi / 180.0) + np.pi)  % (2 * np.pi) - np.pi},
			# 			#'resample': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST,
			# 			#					InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST)  + ((InterpolationMode.BILINEAR, ) if USE_DEPTH else ()),
			# 			'angles': list(range(0,360,10)),
			# 			'rate':0.5,
			# 		}
			# 	},
			# 	{
			# 		'name': 'ColorJitter',
			# 		'opts': {
			# 			'keys': ('image',), 'p': 0.5,
			# 			'saturation': 0.3, 'hue': 0.3, 'brightness': 0.3, 'contrast':0.3
			# 		}
			# 	}
        ]

        subfolders_train = [
            dict(folder='bg=white_desk', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
			dict(folder='bg=green_checkered', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
			dict(folder='bg=poster', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
			dict(folder='bg=red_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
        ]

        subfolders_test = [
            dict(folder='bg=red_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=checkered_rag_small', 'cloth=cotton_napkin']),
            dict(folder='bg=white_desk', annotation='annotations_propagated.xml', data_subfolders=['cloth=checkered_rag_small', 'cloth=cotton_napkin']),
            dict(folder='bg=green_checkered', annotation='annotations_propagated.xml', data_subfolders=['cloth=checkered_rag_small', 'cloth=cotton_napkin']),
            dict(folder='bg=poster', annotation='annotations_propagated.xml', data_subfolders=['cloth=checkered_rag_small', 'cloth=cotton_napkin']),
            dict(folder='bg=festive_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=checkered_rag_small', 'cloth=cotton_napkin', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
        ]
        #subfolders = [dict(folder='../demo_cube', annotation='annotations.json', data_subfolders=['.']),]
        #subfolders = [dict(folder='../IJS-examples',data_subfolders=['.'])]

        #/storage/datasets/ClothDataset
        #CLOTH_DATASET_VICOS = '/storage/local/ssd/cache/ClothDatasetVICOS/'
        CLOTH_DATASET_VICOS = os.environ.get('VICOS_TOWEL_DATASET')
        db = ClothDataset(root_dir=CLOTH_DATASET_VICOS, resize_factor=1, transform_only_valid_centers=1.0, transform_per_sample_rng=False,
                          transform=transform, segment_cloth=USE_SEGMENTATION, use_depth=USE_DEPTH, correct_depth_rotation=False, subfolders=subfolders_train if EVAL_TYPE == "train" else subfolders_test)

        db = KeypointPreprocessorDataset(preprocessor, db, returned_raw_sample=True, randomize_keypoints_order=False)
        # prepare training data
        train_imgs = []
        train_prompts = []

        plt.figure()

        from utils.evaluation.center_eval import CenterGlobalMinimizationEval
        eval = CenterGlobalMinimizationEval("")

        results = dict()

        for i,preeprocessed_sample in enumerate(tqdm(db)):
            #if i % 8 != 0:
            #    continue
            sample = preeprocessed_sample['sample']
            img = np.array(sample['image'])
            center = sample['center']

            gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]
            #gt_centers = gt_centers[:1,:] # only one

            #gt_centers_text = centers_to_tokens(gt_centers, img.shape[1:])
            #gt_kps = extract_individual_keypoints(gt_centers_text, image_info=None)
            #gt_kps[:,0] *= img.shape[1]/config.IMAGE_INPUT_SIZE[0]
            #gt_kps[:,1] *= img.shape[2]/config.IMAGE_INPUT_SIZE[0]

            gt_centers_text = preprocessor.tokenizer.decode(preeprocessed_sample['/targets/text/targets'])
            gt_kps = extract_individual_keypoints(gt_centers_text, preeprocessed_sample["/meta/image_info"])

            train_prompts.append(f"List coordinates of all visible towel corners in <image_input>: {gt_centers_text}")
            train_imgs.append(img)


            kps, text = detect_keypoints(runner, np.transpose(img,(1,2,0)),
                                         generation_config = GenerationConfig(
                                                                do_sample=True,
                                                                num_beams=5,
                                                                max_length=None,  # Avoid warning about preferring max_new_tokens
                                                                bos_token_id=0,
                                                                eos_token_id=1,
                                                                # We generally use 0 for padding, but having pad==bos triggers a superfluous
                                                                # warning from GenerationMixin so we just tell it 1 to keep it quiet
                                                                pad_token_id=1,
                                                            ))

            results[sample['im_name'].replace(CLOTH_DATASET_VICOS,"")] = kps

            if PLOT:
                print(sample['im_name'])
                print("gt:", train_prompts[-1])
                print(gt_centers)
                print("prediction:", text)
                print(kps)

                import PIL
                
                plt.clf()
                plt.subplot(1, 1, 1)            
                plt.imshow(np.transpose(img,(1,2,0)))
                plt.plot(gt_kps[:,0],gt_kps[:,1],'r.')
                plt.plot(kps[:,0],kps[:,1],'bx')
                #plt.plot(kps[:,1]+420,kps[:,0]-420,'bx')
                
                plt.draw(); plt.pause(0.01)
                plt.waitforbuttonpress()	
        
        os.makedirs(os.path.join(EVAL_FOLDER,f"{EVAL_TYPE}_results{EVAL_EPOCH}"), exist_ok=True)

        import pickle
        with open(os.path.join(EVAL_FOLDER,f"{EVAL_TYPE}_results{EVAL_EPOCH}","results.pkl"), 'wb') as f:
            pickle.dump(results, f)


    if False:
        img_filename = "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls4_camera0.jpg"
        #img_filename = "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view19_ls1_camera0.jpg"

        #detect_towel_corners_in_context(runner, ["/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls3_camera0.jpg",
        #                                         "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls5_camera0.jpg",
        #                                         "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls6_camera0.jpg",
        #                                         "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls0_camera0.jpg"], img_filename)
        
        kps, text = detect_keypoints(runner, img_filename)
        print(text)
        print(kps)

        import PIL
        plt.figure()
        plt.imshow(PIL.Image.open(img_filename))
        plt.plot(kps[:,0],kps[:,1],'.')
        plt.show(block=True)

