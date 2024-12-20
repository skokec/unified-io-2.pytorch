import torch

from uio2.model import UnifiedIOModel
from uio2.preprocessing import UnifiedIOPreprocessor
from uio2.runner import TaskRunner, extract_individual_keypoints
from uio2.data_utils import values_to_tokens
from uio2 import config

from transformers import GenerationConfig

import pylab as plt
from PIL import Image
from tqdm import tqdm
import os

from config import get_config_args

import numpy as np

def detect_keypoints(runner, img_filename, image_processed_size, **kwargs):
    p = "List coordinates of all visible towel corners"
    example = runner.uio2_preprocessor(text_inputs=p, image_inputs=img_filename, target_modality="text")
    
    kps, scores, text = runner.keypoint_list_with_scores(example, image_processed_size,  max_tokens=32, **kwargs)
    
    return kps, scores, text


def centers_to_tokens(gt_centers, img_shape):
    
    gt_centers_tokens_x = values_to_tokens(gt_centers[:,0] / img_shape[1])
    gt_centers_tokens_y = values_to_tokens(gt_centers[:,1] / img_shape[0])


    gt_centers_tokens_x = [x.decode("utf-8") for x in gt_centers_tokens_x.numpy()]
    gt_centers_tokens_y = [y.decode("utf-8") for y in gt_centers_tokens_y.numpy()]

    gt_centers_text = " , ".join([f"{x} {y}" for x,y in zip(gt_centers_tokens_x, gt_centers_tokens_y)])

    return gt_centers_text


def parse_img_size_arg(val):
    if 'x' in EVAL_PRERESIZE:
        h, w = map(int,val.split("x"))
    else:
        h = w = int(val)
    return h, w


if __name__ == "__main__":

    args = get_config_args()

    if args.get('attach_debug'):
        import ptvsd

        # Allow other computers to attach to the debugger
        ptvsd.enable_attach(address=('0.0.0.0', args['attach_debug']))
        print(f"Waiting for debugger attach at port {args['attach_debug']}...")
        ptvsd.wait_for_attach()


    PLOT = False

    #EXP_FOLDER = "./exp/vicos-towels/model=allenai/uio2-large_resize_factor=1_batchsize=24/num_train_epoch=100/depth=False"
    #EXP_FOLDER = "./exp/vicos-towels/model=allenai/uio2-large_resize_factor=1_batchsize=24_lr=0.001/num_train_epoch=100/depth=False"
    EVAL_FOLDER = args['save_dir']
    EVAL_TYPE = args['eval_type']
    EVAL_EPOCH = args['eval_epoch']
    EVAL_CROPPED = args.get('eval_cropped')
    EVAL_SW_STEP = args.get('eval_sliding_window_step')
    SKIP_IF_EXISTS = args.get('skip_if_exists')
    EVAL_PRERESIZE = args.get('eval_preresize')
    
    if EVAL_PRERESIZE:
        TEST_SIZE_H, TEST_SIZE_W = parse_img_size_arg(EVAL_PRERESIZE)

    if EVAL_SW_STEP:
        TEST_STEP_H, TEST_STEP_W = parse_img_size_arg(EVAL_SW_STEP)

    DISPLAY_TO_FILE = args.get("display_to_file")
    
    OUTPUT_RESULT_DIR = os.path.join(EVAL_FOLDER,f"{EVAL_TYPE}_results{EVAL_EPOCH}")

    model_cfg_overrides = args['model'].get('kwargs')
    preprocessor_kwargs = args['model'].get('preprocessor_kwargs')

    processing_size = args['model'].get('processing_size')
    if processing_size:
        if model_cfg_overrides is None:
            model_cfg_overrides = dict()
            
        if 't5_config' not in model_cfg_overrides:
            model_cfg_overrides['t5_config'] = dict()
        if 'sequence_length' not in model_cfg_overrides:
            model_cfg_overrides['sequence_length'] = dict()

        from uio2 import config

        model_cfg_overrides['t5_config']['default_image_vit_size'] = tuple(processing_size)
        #model_cfg_overrides['t5_config']['encoder_max_image_length'] = (processing_size[0]//config.IMAGE_INPUT_D)*(processing_size[1]//config.IMAGE_INPUT_D)
        model_cfg_overrides['sequence_length']['image_input_samples'] = (processing_size[0]//config.IMAGE_INPUT_D)*(processing_size[1]//config.IMAGE_INPUT_D)
        

        if 'cfg_overrides' not in preprocessor_kwargs:
            preprocessor_kwargs['cfg_overrides'] = dict()
        if 'sequence_length' not in preprocessor_kwargs['cfg_overrides']:
            preprocessor_kwargs['cfg_overrides']['sequence_length'] = dict()
        if 't5_config' not in preprocessor_kwargs['cfg_overrides']:
            preprocessor_kwargs['cfg_overrides']['t5_config'] = dict()

        preprocessor_kwargs['cfg_overrides']['t5_config']['default_image_vit_size'] = tuple(processing_size)
        #preprocessor_kwargs['cfg_overrides']['t5_config']['encoder_max_image_length'] = (processing_size[0]//config.IMAGE_INPUT_D)*(processing_size[1]//config.IMAGE_INPUT_D)
        preprocessor_kwargs['cfg_overrides']['sequence_length']['image_input_samples'] = (processing_size[0]//config.IMAGE_INPUT_D)*(processing_size[1]//config.IMAGE_INPUT_D)

        OUTPUT_RESULT_DIR = os.path.join(OUTPUT_RESULT_DIR, f"processing_size={processing_size[0]}x{processing_size[1]}")

    if EVAL_PRERESIZE:
        OUTPUT_RESULT_DIR = os.path.join(OUTPUT_RESULT_DIR, f"preresize_size={TEST_SIZE_H}x{TEST_SIZE_W}")

    if EVAL_SW_STEP:
        assert EVAL_PRERESIZE, "Missing 'eval_preresize' when enabling sliding window!"

        OUTPUT_RESULT_DIR = os.path.join(OUTPUT_RESULT_DIR, f"sliding_window_step={TEST_STEP_H}x{TEST_STEP_W}")

    if EVAL_CROPPED:
        OUTPUT_RESULT = os.path.join(OUTPUT_RESULT_DIR, "results_cropped_img.pkl")
    else:
        OUTPUT_RESULT = os.path.join(OUTPUT_RESULT_DIR, "results.pkl")

    if SKIP_IF_EXISTS and os.path.exists(OUTPUT_RESULT):
        print(f"Skipping due to found existing results {OUTPUT_RESULT}")
    else:
        dev = torch.device("cuda:0")

        preprocessor = UnifiedIOPreprocessor.from_pretrained(args['model']['preprocessor'], **preprocessor_kwargs)
        model = UnifiedIOModel.from_pretrained(args['model']['name'], cfg_overrides=model_cfg_overrides, local_files_only=True)

        
        state = torch.load(os.path.join(EVAL_FOLDER,f"checkpoint{EVAL_EPOCH}.pth"))

        model_state_dict = {k.replace("module.","").replace(".out.",".out_proj."):v for k,v in state['model_state_dict'].items()}
        model.load_state_dict(model_state_dict, strict=True)
        model.to(dev)

        model.eval()

        runner = TaskRunner(model, preprocessor)

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

        if EVAL_CROPPED:
            RESIZE_FACTOR=0.5
            transform.append(
                {
                'name': 'RandomCrop',
                'opts': {
                    'keys': ('image',),'keys_bbox': ('center',),
                    'pad_if_needed': True,
                    'size': (512,512)
                }
                })
        else:
            #RESIZE_FACTOR=1.0
            RESIZE_FACTOR=0.5

            if EVAL_PRERESIZE:
                assert TEST_SIZE_W > 0 and TEST_SIZE_W > 0, "Invalid values for eval_preresize"
                
                transform.append(
                    {
                        'name': 'Resize',
                        'opts': {
                            'keys': ('image',),
                            'interpolation': (InterpolationMode.BILINEAR,),
                            'keys_bbox': ('center',),
                            'size': (TEST_SIZE_H, TEST_SIZE_W),
                        }
                    })
        
        #/storage/datasets/ClothDataset
        #CLOTH_DATASET_VICOS = '/storage/local/ssd/cache/ClothDatasetVICOS/'
        CLOTH_DATASET_VICOS = os.environ.get('VICOS_TOWEL_DATASET')
        db = ClothDataset(root_dir=CLOTH_DATASET_VICOS, resize_factor=RESIZE_FACTOR, transform_only_valid_centers=1.0, transform_per_sample_rng=False,
                          transform=transform, segment_cloth=USE_SEGMENTATION, use_depth=USE_DEPTH, correct_depth_rotation=False, subfolders=subfolders_train if EVAL_TYPE == "train" else subfolders_test,
                          #valid_img_names=['bg=red_tablecloth/cloth=checkered_rag_small/rgb/image_0000_view0_',]
                          )


        if EVAL_SW_STEP:
            from datasets.PatchSplitDataset import PatchSplitDataset
            db = PatchSplitDataset(db, patch_image_size=(TEST_SIZE_W,TEST_SIZE_H), patch_steps=(TEST_STEP_W,TEST_STEP_H))


        db = KeypointPreprocessorDataset(preprocessor, db, full_config=model.full_config, returned_raw_sample=True, randomize_keypoints_order=False, apply_internal_scale_aug=False)
        # prepare training data
        train_imgs = []
        train_prompts = []

        plt.figure()

        from utils.evaluation.center_eval import CenterGlobalMinimizationEval
        eval = CenterGlobalMinimizationEval("")

        results = dict()

        IMAGE_PROCESS_SIZE = model.config.default_image_vit_size if model.full_config.use_image_vit else model.config.default_image_size

        from utils.utils import ImageGridCombiner
        
        grid_combiner = ImageGridCombiner()
        grid_index = None

        def merge_patch_results(data):

            predictions_merge_thr=0.5
            predictions_merge_dist_thr=10
    
            im_name = data.get_image_name()
            im_shape = data.full_size

            kps, _, _ = data.get_instance_description('predictions', im_shape, out_mask_tensor=None,
                                                        overlap_thr=predictions_merge_thr,
                                                        merge_dist_thr=predictions_merge_dist_thr)
            if len(kps) > 0:
                num_cols = kps[list(kps.keys())[0]].shape[0]
                kps = np.array([kps[k] if k in kps else np.zeros((num_cols,)) for k in range(max(kps.keys())+1) if k > 0])
            else:
                kps = np.zeros((0,4))

            return kps, im_name


        for i,preeprocessed_sample in enumerate(tqdm(db)):
            
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
            gt_kps = extract_individual_keypoints(gt_centers_text, preeprocessed_sample["/meta/image_info"], IMAGE_PROCESS_SIZE)

            train_prompts.append(f"List coordinates of all visible towel corners in <image_input>: {gt_centers_text}")
            train_imgs.append(img)

            kps, scores, text = detect_keypoints(runner, np.transpose(img,(1,2,0)), image_processed_size=IMAGE_PROCESS_SIZE,
                                                 generation_config = GenerationConfig(                                                                
                                                                #do_sample=True,
                                                                #num_beams=5,
                                                                max_length=None,  # Avoid warning about preferring max_new_tokens
                                                                bos_token_id=0,
                                                                eos_token_id=1,
                                                                # We generally use 0 for padding, but having pad==bos triggers a superfluous
                                                                # warning from GenerationMixin so we just tell it 1 to keep it quiet
                                                                pad_token_id=1,
                                                            ))
            
            im_name = sample['im_name']

            if PLOT:
                print(im_name)
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
            
            if DISPLAY_TO_FILE:
                import cv2
                img_denormalized = (img * 255).astype(np.uint8)
                img_transposed = np.transpose(img_denormalized, (1, 2, 0))
                # Convert the image to BGR (OpenCV uses BGR, while matplotlib uses RGB)
                img_bgr = cv2.cvtColor(img_transposed, cv2.COLOR_RGB2BGR)

                # Clone the image for drawing keypoints
                img_copy = img_bgr.copy()

                # Draw ground truth keypoints (red dots)
                for pt in gt_kps:
                    cv2.circle(img_copy, (int(pt[0]), int(pt[1])), radius=5, color=(0, 0, 255), thickness=-1)

                # Draw predicted keypoints (blue x's)
                for pt in kps:
                    cv2.drawMarker(img_copy, (int(pt[0]), int(pt[1])), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

                os.makedirs(os.path.join(os.path.dirname(OUTPUT_RESULT),"plot_cropped" if EVAL_CROPPED else "plot"), exist_ok=True)
                cv2.imwrite(os.path.join(os.path.dirname(OUTPUT_RESULT),"plot_cropped" if EVAL_CROPPED else "plot", im_name.replace(CLOTH_DATASET_VICOS,"").replace("/",".")), img_copy)

            if 'RandomCrop' in sample:
                params, pad_size = sample['RandomCrop']
                # pad_size = (t,l,r,b) -- unclear if top and left are switched
                # params = (dx,dy, th, tw)
                kps = kps.reshape(-1,2)
                
                kps[:,0] = (kps[:,0] + params[1] - pad_size[0])
                kps[:,1] = (kps[:,1] + params[0] - pad_size[1])

            if 'Resize' in sample:
                resize_factors = sample['Resize']

                kps = kps.reshape(-1,2)
                
                kps[:,0] = (kps[:,0]/ resize_factors[0] )
                kps[:,1] = (kps[:,1]/ resize_factors[1] )


            if 'grid_index' in sample:
                grid_index = sample['grid_index']

                pred = {i + 1: np.concatenate([pred, scores[i]],axis=0) for i, pred in enumerate(kps)}
                pred_mask = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.uint8)


                M = 2
                for i,p in pred.items():
                    pred_mask[int(p[1] - M):int(p[1] + M), int(p[0] - M):int(p[0] + M)] = i

                # send data to grid_combiner for merging
                # when new index is detected it will merge and return previously collected data
                finished_image = grid_combiner.add_image(sample['im_name'],
                                                         grid_index,
                                                         data_map_tensor={},
                                                         data_map_instance_desc=dict(predictions=(pred, pred_mask, False)))

                # if this is final image then extract merged result and return it
                if finished_image is not None:
                    kps, im_name = merge_patch_results(finished_image)                   
                    kps, scores = kps[:,:2], kps[:,2:]
                else:
                    continue
                        
            results[im_name.replace(CLOTH_DATASET_VICOS,"")] = np.concatenate((kps/RESIZE_FACTOR, scores),axis=1)

        # need to additionally handle last sample when in patch-split
        if grid_index is not None:
            kps, im_name = merge_patch_results(grid_combiner.current_data)
            kps, scores = kps[:,:2], kps[:,2:]

            results[im_name.replace(CLOTH_DATASET_VICOS,"")] = np.concatenate((kps/RESIZE_FACTOR, scores),axis=1)

        
        os.makedirs(os.path.dirname(OUTPUT_RESULT), exist_ok=True)

        import pickle
        with open(OUTPUT_RESULT, 'wb') as f:
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

