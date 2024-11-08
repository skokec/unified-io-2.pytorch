import glob
import os, cv2, sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import json
import torch
from torch.utils.data import Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import transforms as my_transforms

class MuJoCoDataset(Dataset):

	def __init__(self, root_dir='./', subfolder="", MAX_NUM_CENTERS=1024, 
			  	transform=None, transform_only_valid_centers=False, transform_per_sample_rng=False,  
				use_depth=False, segment_cloth=False, use_normals=False, fixed_bbox_size=15, resize_factor=1, 
				num_cpu_threads=1, normals_mode=1, reference_normal = [0,0,1], valid_img_names=None, **kwargs):
		print('MuJoCo Dataset created')

		if num_cpu_threads:
			torch.set_num_threads(num_cpu_threads)

		self.root_dir = root_dir
		self.MAX_NUM_CENTERS = MAX_NUM_CENTERS		
		self.use_depth = use_depth
		self.use_normals = use_normals
		self.fixed_bbox_size = fixed_bbox_size
		self.resize_factor = resize_factor
		self.normals_mode = normals_mode
		self.reference_normal = reference_normal
		self.segment_cloth = segment_cloth

		self.transform = transform
		self.rng = np.random.default_rng(1337)
		self.transform_only_valid_centers = transform_only_valid_centers
		self.transform_per_sample_rng = transform_per_sample_rng
		self.return_image = True

		print("dataset use_depth", use_depth)

		print("root_dir", root_dir)
		print("subfolder", subfolder)
		print("segmentation", segment_cloth)
		if type(subfolder) not in [list, tuple]:
			subfolder = [subfolder]

		image_list = []
		for sub in subfolder:
			image_list += sorted(glob.glob(f"{self.root_dir}/{sub}/rgb/*"))

		if valid_img_names is not None:		
			def filter_by_name(x):				
				return any([v in x for v in valid_img_names])
			
			image_list = list(filter(filter_by_name, image_list))


		self.image_list = image_list
		print(f'MuJoCoDataset of size {len(image_list)}')

		self.size = len(self.image_list)

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		im_fn = self.image_list[index]

		fn = os.path.splitext(os.path.split(im_fn)[-1])[0]	

		image = Image.open(im_fn)
		org_im_size = np.array(image.size)

		if self.resize_factor is not None:
			im_size = int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor)

		# avoid loading full buffer data if image not requested
		if self.return_image:
			if self.resize_factor is not None and self.resize_factor != 1.0:
				image = image.resize(im_size, Image.BILINEAR)
		else:
			image = None

		root_dir = os.path.abspath(os.path.join(os.path.dirname(im_fn),'..'))
		depth_fn = os.path.join(root_dir, 'depth', f'{fn}.npy')

		gt_fn = os.path.join(root_dir, 'gt_points_vectors', f'{fn}.npy')
		if os.path.exists(gt_fn):
			gt_data = np.load(gt_fn)
		else:
			print(gt_fn, "not found")
			gt_data = []

		sample = dict(
			image=image,
			im_name=im_fn,
			im_size=im_size,
			org_im_size=org_im_size,
			index=index,
		)	

		if self.segment_cloth:
			gt_seg_fn = os.path.join(root_dir, "gt_cloth", f"{fn}.png")

			segmentation_mask = Image.open(gt_seg_fn)

			if self.resize_factor is not None and self.resize_factor != 1.0:
				segmentation_mask = segmentation_mask.resize(im_size, Image.BILINEAR)

			sample["segmentation_mask"] = segmentation_mask


		centers = []

		for n, (i,j,s,c) in enumerate(gt_data):
			i = int(i)
			j = int(j)
			centers.append((i,j))

		centers = np.array(centers)
		sample['center'] = np.zeros((self.MAX_NUM_CENTERS, 2))
		try:
			sample['center'][:centers.shape[0], :] = centers
		except:
			print("no objects in image")

		if self.transform is not None:		
			transform = my_transforms.get_transform(self.transform) if self.transform is not None and type(self.transform) == list else transform	
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
			sample['image'] = torch.cat((sample['image'], sample['depth']))

		return sample
	


if __name__ == "__main__":
	import pylab as plt
	import matplotlib

	matplotlib.use('TkAgg')
	from tqdm import tqdm
	import torch

	USE_DEPTH = True
	from utils import transforms as my_transforms

	transform = my_transforms.get_transform([
		{
			'name': 'ToTensor',
			'opts': {
				'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
				'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor,
						 torch.ByteTensor)+ ((torch.FloatTensor, ) if USE_DEPTH else ()),
			},			
		}
	])
	subfolders = ['mujoco', 'mujoco_all_combinations_normal_color_temp', 'mujoco_all_combinations_rgb_light', 'mujoco_white_desk_HS_extreme_color_temp', 'mujoco_white_desk_HS_normal_color_temp']

	db = MuJoCoDataset(root_dir='/storage/datasets/ClothDataset/', resize_factor=1, transform_only_valid_centers=1.0, transform=transform, use_depth=USE_DEPTH, correct_depth_rotation=False, subfolder=subfolders)
	shapes = []
	for item in tqdm(db):
	#for item in db:
		#continue
		if item['index'] % 50 == 0:
			print('loaded index %d' % item['index'])
		shapes.append(item['image'].shape)
		# if True or np.array(item['ignore']).sum() > 0:
		# if True:
		if item['index'] % 1 == 0:
			center = item['center']
			gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]
			# print(gt_centers)
			plt.clf()

			im = item['image'].permute([1, 2, 0]).numpy()
			# print(im.shape)

			plt.subplot(2, 2, 1)
			plt.imshow(im[...,:3])
			plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')

			x = gt_centers[:,0]
			y = gt_centers[:,1]

			r = 100

			for i,j in zip(x,y):
				i = int(i)
				j = int(j)
				if i < 0 or i > item['orientation'].shape[2] or \
					j < 0 or j > item['orientation'].shape[1]:
					continue
				angle = item['orientation'][0][j,i].numpy()
				# print(angle)
				# s = item['orientation'][1][j,i]
				s = -np.sin(angle)
				c = -np.cos(angle)
				# print(i,j,c,s)
				plt.plot([i,i+r*s],[j,j+r*c], 'r-')



			plt.draw(); plt.pause(0.01)
			plt.waitforbuttonpress()
			# plt.show()

	print("end")
