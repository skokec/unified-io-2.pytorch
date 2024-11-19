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

args = dict(
	cuda=True,

	tf_logging=['loss'],
	tf_logging_iter=2,

	save=True,
	save_interval=5,

	# --------
	n_epochs=10,

	extra_str='',

	save_dir=os.path.join(OUTPUT_DIR, 'mujoco',
						  'model={args[model][name]}_resize_factor=1.0_batchsize={args[train_dataset][batch_size]}_lr={args[model][lr]}_with_prompt_and_gt_randomization{args[extra_str]}',
						  'num_train_epoch={args[n_epochs]}', 
						  f'depth={USE_DEPTH}',
                          ),

	pretrained_model_path = None,
	resume_path = None,


	train_dataset = {
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
			'transform': [
				{
					'name': 'ToTensor',
					'opts': {
						'keys': ('image','segmentation_mask', 'edge_mask', 'outer_edge_mask', 'inner_edge_mask'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor),
					}
				},
				{
					'name': 'RandomHorizontalFlip',
					'opts': {
                        'keys': ('image','segmentation_mask', 'edge_mask', 'outer_edge_mask', 'inner_edge_mask'),'keys_bbox': ('center',),
						'p': 0.5,
					}
				},
				{
					'name': 'RandomVerticalFlip',
					'opts': {
						'keys': ('image','segmentation_mask', 'edge_mask', 'outer_edge_mask', 'inner_edge_mask'),'keys_bbox': ('center',),
						'p': 0.5,
					}
				},
				{
					'name': 'RandomCustomRotation',
					'opts': {
						'keys': ('image','segmentation_mask', 'edge_mask', 'outer_edge_mask', 'inner_edge_mask'),'keys_bbox': ('center',),
						'resample': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
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
			'MAX_NUM_CENTERS': 128
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
