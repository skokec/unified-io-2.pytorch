import copy
import os

import torch
from utils import transforms as my_transforms
from torchvision.transforms import InterpolationMode

LLAMA2_TOKENIZER = os.environ.get('LLAMA2_TOKENIZER')
CLOTH_DATASET_VICOS = os.environ.get('VICOS_TOWEL_DATASET')
MUJOCO_DATASET_VICOS = os.environ.get('VICOS_MUJOCO_DATASET')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR','./exp')
USE_DEPTH = False

import numpy as np

def rotate_orientation_values(orientation, angle):
	# every value in orientation (in radians) should be rotated by provided angle (in degrees) and returned	
    return ((orientation + (angle*np.pi / 180.0)) + np.pi) % (2 * np.pi) - np.pi

transforms = [{
		'name': 'ToTensor',
		'opts': {
			'keys': ('image',),
			'type': (torch.FloatTensor,),
		}
	},
	{
		'name': 'RandomHorizontalFlip',
		'opts': {
			'keys': ('image',),'keys_bbox': ('center',),
			'p': 0.5,
		}
	},
	{
		'name': 'RandomVerticalFlip',
		'opts': {
			'keys': ('image',),'keys_bbox': ('center',),
			'p': 0.5,
		}
	},
	{
		'name': 'RandomCustomRotation',
		'opts': {
			'keys': ('image',),'keys_bbox': ('center',),
			'resample': (InterpolationMode.BILINEAR,),
			'angles': list(range(0,360,10)),
			'rate':0.5,
		}
	},
	{
		'name': 'RandomCrop',
		'opts': {
			'keys': ('image',),'keys_bbox': ('center',),
			'pad_if_needed': True,
			'size': (512,512)
		}
	},
	{
		'name': 'ColorJitter',
		'opts': {
			'keys': ('image',), 'p': 0.5,
			'saturation': 0.3, 'hue': 0.3, 'brightness': 0.3, 'contrast':0.3
		}
	}]

RESIZE_FACTOR=0.5

args = dict(
	cuda=True,

	tf_logging=['loss'],
	tf_logging_iter=2,

	save=True,
	save_interval=5,

	# --------
	n_epochs=10,

	extra_str='',

	save_dir=os.path.join(OUTPUT_DIR, 'vicos-towels+mujoco',
						  'model={args[model][name]}_resize_factor=%s_batchsize={args[train_dataset][batch_size]}_lr={args[model][lr]}_with_prompt_and_gt_randomization{args[extra_str]}' % str(RESIZE_FACTOR),
						  'num_train_epoch={args[n_epochs]}', 
						  f'depth={USE_DEPTH}',
                          ),

	pretrained_model_path = None,
	resume_path = None,


	train_dataset = {
        'name': 'concat',
        'kwargs': [{
			'name': 'vicos-towel',
			'kwargs': {
				'normalize': False,
				'root_dir': os.path.abspath(CLOTH_DATASET_VICOS),
				'subfolders': [dict(folder='bg=white_desk', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
							dict(folder='bg=green_checkered', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
							dict(folder='bg=poster', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
							dict(folder='bg=red_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
							],
				'resize_factor': RESIZE_FACTOR,
				'use_depth': USE_DEPTH,
				'correct_depth_rotation': False,
				'use_mean_for_depth_nan': True,
				'use_normals': False,
				'transform_per_sample_rng': False,
				'transform_only_valid_centers': 1.0, # all points must still be visible after transform
				'transform': transforms,
				'MAX_NUM_CENTERS': 128
			},
		}, {
			'name': 'mujoco',
			'kwargs': {
				'normalize': False,
				'root_dir': MUJOCO_DATASET_VICOS,
				'subfolder': ['mujoco',
							'mujoco_all_combinations_normal_color_temp',
							'mujoco_all_combinations_rgb_light',
							'mujoco_white_desk_HS_extreme_color_temp',
							'mujoco_white_desk_HS_normal_color_temp'],
				'use_depth': USE_DEPTH,
				'correct_depth_rotation': False,
				'use_mean_for_depth_nan': True,
				'use_normals': False,
				'transform_per_sample_rng': False,
				'transform_only_valid_centers': 1.0, # all points must still be visible after transform
				'transform': transforms,
				'MAX_NUM_CENTERS': 128
			},
		}
        ],

		'batch_size': 2,
		'workers': 4,
        'force_workers_on_distributed_processing': True,
		'shuffle': True,
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
