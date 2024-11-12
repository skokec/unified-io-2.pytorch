import copy
import os

import torch
from utils import transforms as my_transforms
from torchvision.transforms import InterpolationMode

LLAMA2_TOKENIZER = os.environ.get('LLAMA2_TOKENIZER')
CLOTH_DATASET_VICOS = os.environ.get('VICOS_TOWEL_DATASET')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR','./exp')
USE_DEPTH = False

import numpy as np

def rotate_orientation_values(orientation, angle):
	# every value in orientation (in radians) should be rotated by provided angle (in degrees) and returned	
    return ((orientation + (angle*np.pi / 180.0)) + np.pi) % (2 * np.pi) - np.pi


args = dict(
	cuda=True,

	tf_logging=['loss'],
	tf_logging_iter=2,

	save=True,
	save_interval=5,

	# --------
	n_epochs=10,

	extra_str='',

	save_dir=os.path.join(OUTPUT_DIR, 'vicos-towels',
						  'model={args[model][name]}_resize_factor={args[train_dataset][kwargs][resize_factor]}_batchsize={args[train_dataset][batch_size]}_lr={args[model][lr]}_with_prompt_and_gt_randomization{args[extra_str]}',
						  'num_train_epoch={args[n_epochs]}', 
						  'depth={args[train_dataset][kwargs][use_depth]}',
                          ),

	pretrained_model_path = None,
	resume_path = None,


	train_dataset = {
		'name': 'vicos-towel',
		'kwargs': {
			'normalize': False,
			'root_dir': os.path.abspath(CLOTH_DATASET_VICOS),
			'subfolders': [dict(folder='bg=white_desk', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
						   dict(folder='bg=green_checkered', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
						   dict(folder='bg=poster', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
						   dict(folder='bg=red_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
						   ],
			'fixed_bbox_size': 15,
			'resize_factor': 0.5,
			'use_depth': USE_DEPTH,
			'correct_depth_rotation': False,
			'use_mean_for_depth_nan': True,
			'use_normals': False,
			'transform_per_sample_rng': False,
            'transform_only_valid_centers': 1.0, # all points must still be visible after transform
			'transform': [
				# for training without augmentation (same as testing)
				{
					'name': 'ToTensor',
					'opts': {
						'keys': ('image',),
						#'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
						#'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor) + ((torch.FloatTensor, ) if USE_DEPTH else ()),
                        'type': (torch.FloatTensor,),
					}
				},
				{
					'name': 'RandomHorizontalFlip',
					'opts': {
                        'keys': ('image',),'keys_bbox': ('center',),
						#'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
						#'keys_custom_fn' : { 'orientation': lambda x: (-1*x + np.pi)  % (2 * np.pi) - np.pi},
						'p': 0.5,
					}
				},
				{
					'name': 'RandomVerticalFlip',
					'opts': {
						'keys': ('image',),'keys_bbox': ('center',),
						#'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
						#'keys_custom_fn' : { 'orientation': lambda x: (np.pi - x + np.pi)  % (2 * np.pi) - np.pi},
						'p': 0.5,
					}
				},
				{
					'name': 'RandomCustomRotation',
					'opts': {
						'keys': ('image',),'keys_bbox': ('center',),
						'resample': (InterpolationMode.BILINEAR,),
						#'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
						#'keys_custom_fn' : { 'orientation': lambda x,angle: (x + (angle*np.pi / 180.0) + np.pi)  % (2 * np.pi) - np.pi},
						#'resample': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST,
						#					InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST)  + ((InterpolationMode.BILINEAR, ) if USE_DEPTH else ()),
						'angles': list(range(0,360,10)),
						'rate':0.5,
					}
				},
				{
					'name': 'ColorJitter',
					'opts': {
						'keys': ('image',), 'p': 0.5,
						'saturation': 0.3, 'hue': 0.3, 'brightness': 0.3, 'contrast':0.3
					}
				}

			],
		},

		'batch_size': 2,
		'workers': 4,
        'force_workers_on_distributed_processing': True,
		'shuffle': True,
        'keypoint_preprocesser_kwargs': {
              'randomize_keypoints_order': True,
              'jitter_keypoints_px': False
		}
	}, 

	model = dict(
		name='allenai/uio2-large', 
        #name='allenai/uio2-large-bfloat16',
        preprocessor='allenai/uio2-preprocessor',
        preprocessor_kwargs=dict(tokenizer=LLAMA2_TOKENIZER),
		optimizer='Adam',
        #lambda_scheduler_fn=lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9)),
        lambda_scheduler_fn=lambda _args: (lambda epoch: 1.0), # disabled
		lr=1e-4,
		weight_decay=0,

	),
)

def get_args():
	return copy.deepcopy(args)
