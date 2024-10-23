import glob
import os, cv2

import numpy as np
from PIL import Image

import json,  os

import torch
from torch.utils.data import Dataset

from utils import transforms as my_transforms
class ClothDataset(Dataset):
	IGNORE_LABEL = "ignore-region"
    
	IGNORE_FLAG = 1
	IGNORE_TRUNCATED_FLAG = 2
	IGNORE_OVERLAP_BORDER_FLAG = 4
	IGNORE_DIFFICULT_FLAG = 8
    
	def __init__(self, root_dir='./', subfolders=None, fixed_bbox_size=15, resize_factor=None, use_depth=False, segment_cloth=False, MAX_NUM_CENTERS=1024, 
			  	valid_img_names=None, num_cpu_threads=1, use_mean_for_depth_nan=False, 
				use_normals=False, normals_mode=1, use_only_depth=False, correct_depth_rotation=False,
				check_consistency=True, transform=None, transform_only_valid_centers=False, transform_per_sample_rng=False, **kwargs):
		print('ViCoS ClothDataset created')

		if num_cpu_threads:
			torch.set_num_threads(num_cpu_threads)

		self.use_depth = use_depth
		self.use_mean_for_depth_nan = use_mean_for_depth_nan
		self.use_normals = use_normals
		self.normals_mode = normals_mode
		self.use_only_depth = use_only_depth
		self.segment_cloth = segment_cloth

		self.correct_depth_rotation = correct_depth_rotation

		self.fixed_bbox_size = fixed_bbox_size
		self.resize_factor = resize_factor

		self.MAX_NUM_CENTERS = MAX_NUM_CENTERS

		self.return_image = True
		self.check_consistency = check_consistency

		self.transform = transform
		self.rng = np.random.default_rng(1337)
		self.transform_only_valid_centers = transform_only_valid_centers
		self.transform_per_sample_rng = transform_per_sample_rng

		self.remove_out_of_bounds_centers = True
		self.BORDER_MARGIN_FOR_CENTER = 1

		# calibration parameters
		fx = 1081.3720703125
		cx = 959.5
		cy = 539.5

		self.K = np.array([[fx, 0.0, cx], [0, fx, cy], [0,0,1]])

		# find images

		image_list = []
		annot = {}

		if subfolders is None:
			subfolders = ["*/*"]
		if type(subfolders) not in [list, tuple]:
			subfolders = [subfolders]

		# expecting annotations in root if available
		annotations_list = {root_dir: 'annotations.json'}

		for sub in subfolders:
			if type(sub) is dict:
				sub_name = sub['folder']								
				data_paths = sub['data_subfolders']

				for data_path in data_paths:
					image_list += sorted(glob.glob(os.path.join(root_dir, sub_name, data_path, 'rgb', '*')))

				# if subfolder has its own anotations use them
				annotations_list[os.path.join(root_dir,sub_name)] = 'annotations.json'

			else:
				# print(os.path.join(root_dir,sub,'rgb','*'))
				image_list += sorted(glob.glob(os.path.join(root_dir,sub,'rgb','*')))

		# read and merge annotations
		annotations = {}
		for sub,annot_filename in annotations_list.items():
			annot_path = os.path.join(sub,annot_filename)

			if not os.path.exists(annot_path):
				continue

			with open(os.path.join(sub,annot_filename)) as f:
				annot = json.load(f)
			for k,v in annot.items():
				annotations[os.path.abspath(os.path.join(sub,k))] = v

		# for k, v in annotations.items():
		# 	print(k,len(v['points']))
		def is_clutter_and_light3(x):
			lightning = int(x.split('/')[-1].split('_')[3].replace('ls', ''))

			view = int(x.split('/')[-1].split('_')[2].replace('view', ''))
			# configuration and clutter are encoded in view, clutter is on if view is odd
			clutter = 'on' if view % 2 == 1 else 'off'

			return lightning == 3 and clutter == 'on'

		#image_list = list(filter(is_clutter_and_light3, image_list))

		if valid_img_names is not None:		
			def filter_by_name(x):				
				return any([v in x for v in valid_img_names])
			
			image_list = list(filter(filter_by_name, image_list))
		

		self.image_list = image_list
		self.annotations = annotations

		self.size = len(self.image_list)
		print(f'ViCoS ClothDataset of size {len(image_list)}')        

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		im_fn = self.image_list[index]

		root_dir = os.path.abspath(os.path.join(os.path.dirname(im_fn),'..'))
		fn = os.path.splitext(os.path.split(im_fn)[-1])[0]

		ann_key = os.path.abspath(im_fn)

		image = Image.open(im_fn)
		im_size = image.size
		org_im_size = np.array(image.size)

		if self.resize_factor is not None:
			im_size = int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor)

		# avoid loading full buffer data if image not requested
		if self.return_image:
			if self.resize_factor is not None and self.resize_factor != 1.0:
				image = image.resize(im_size, Image.BILINEAR)
		else:
			image = None

		sample = dict(image=image,
					  im_name=im_fn,
					  org_im_size=org_im_size,
					  im_size=im_size,
					  index=index)
		
		if self.segment_cloth:
			segmentation_mask_file = os.path.join(root_dir, "mask", f"{fn}.png")
			segmentation_mask = Image.open(segmentation_mask_file)
			
			if self.resize_factor is not None and self.resize_factor != 1.0:
				segmentation_mask = segmentation_mask.resize(im_size, Image.BILINEAR)

			sample["segmentation_mask"] = segmentation_mask

		if self.use_depth and False:
			from utils.utils_depth import get_normals, eul2rot, rotate_depth
			
			depth_fn = os.path.join(root_dir, 'depth', f'{fn}.npy')
			# print(depth_fn, os.path.exists(depth_fn))
			depth = np.load(depth_fn)
			
			if self.resize_factor is not None and self.resize_factor != 1.0:
				depth = cv2.resize(depth,im_size)
				
			invalid_mask = np.isinf(depth) | np.isnan(depth) | (depth > 1e4) | (depth<0)
			depth[invalid_mask]=depth[~invalid_mask].mean() if self.use_mean_for_depth_nan else 1e-6

			depth*=1e-3 # mm to m

			# correct depth values so the surface is parallel to the image plane
			if self.correct_depth_rotation:
				surface_pitch = self.annotations.get(ann_key)['surface_pitch']
				#print("surface_pitch", surface_pitch)
				R = eul2rot((np.radians(surface_pitch), 0,0))
				depth = rotate_depth(depth, R, self.K)

			if self.use_normals:
				depth = get_normals(depth, normals_mode=self.normals_mode, household=True)
			else:
				depth/=np.max(depth)

			sample['depth'] = depth

		centers = []

		annot = self.annotations.get(ann_key)['points']

		if annot:
			for x1,y1,x2,y2 in annot:

				pt1 = np.array([x1, y1])
				pt2 = np.array([x2, y2])

				if self.resize_factor is not None and self.resize_factor != 1.0:
					pt1 = pt1 * self.resize_factor
					pt2 = pt2 * self.resize_factor

				direction = pt1 - pt2
				direction = np.arctan2(direction[0], direction[1])

				centers.append(pt1)

			centers = np.array(centers)

		sample['center'] = np.zeros((self.MAX_NUM_CENTERS, 2))
		if len(centers) > 0:
			sample['center'][:centers.shape[0], :] = centers
		sample['name'] = im_fn
		sample['image'] = np.array(sample['image'])

		if self.transform is not None:
			transform = my_transforms.get_transform(self.transform) if type(self.transform) == list else self.transform
			import copy
			do_transform = True

			ii = 0
			while do_transform:			
				if ii > 10 and ii % 10 == 0:
					print(f"WARNING: unable to generate valid transform for {ii} iterations")
				new_sample = transform(copy.deepcopy(sample), self.rng if not self.transform_per_sample_rng else np.random.default_rng(1337))

				out_of_bounds_ids = [id for id, c in enumerate(new_sample['center']) if c[0] < 0 or c[1] < 0 or c[0] >= new_sample['image'].shape[-1] or c[1] >= new_sample['image'].shape[-2]]

				# stop if sufficent centers still visible
				if not self.transform_only_valid_centers or self.transform_only_valid_centers <= 0:
					do_transform = False
					sample = new_sample
				else:
					if type(self.transform_only_valid_centers) == bool:
						if len(out_of_bounds_ids) < len(centers): # at least one must be present
							do_transform = False
							sample = new_sample
					elif type(self.transform_only_valid_centers) == int:
						if len(centers) - len(out_of_bounds_ids) >= self.transform_only_valid_centers:
							do_transform = False
							sample = new_sample
					elif type(self.transform_only_valid_centers) == float:
						min_visible = int(self.transform_only_valid_centers * len(centers))
						if len(centers) - len(out_of_bounds_ids) >= min_visible:
							do_transform = False
							sample = new_sample
					else:
						raise Exception("Invalid type of transform_only_valid_centers, allowed types: bool, int, float")				
				ii += 1

		if self.use_depth:
			if self.use_only_depth:
				sample['image'] = torch.cat((sample['image']*0, sample['depth']))
			else:
				sample['image'] = torch.cat((sample['image'], sample['depth']))                     

		if self.remove_out_of_bounds_centers:
			# if instance has out-of-bounds center then ignore it if requested so
			out_of_bounds_ids = [id for id, c in enumerate(sample['center'])
							 # if center closer to border then this margin than mark it as truncated
							 if id >= 0 and (c[0] < 0 or c[1] < 0 or
											c[0] >= sample['image'].shape[-1] or
											c[1] >= sample['image'].shape[-2])]
			for id in out_of_bounds_ids:
				sample['center'][id,:] = -1

		return sample
