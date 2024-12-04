import os, sys
import cv2

import pylab as plt
import numpy as np
from PIL import Image, ImageFile

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import transforms as my_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def from_uint16_to_double_uint8(M):
    return [(M // 256).astype(np.uint8), (M % 256).astype(np.uint8)]


def from_double_uint8_to_uint16(M_upper, M_lower):
    return M_upper.astype(np.uint16) * 256 + M_lower.astype(np.uint16)

def from_double_uint8_to_int16(M_upper, M_lower):
    return M_upper.astype(np.int16) * 256 + M_lower.astype(np.int16)


class PatchSplitDataset(Dataset):
    IGNORE_FLAG = 1
    IGNORE_TRUNCATED_FLAG = 2
    IGNORE_OVERLAP_BORDER_FLAG = 4

    def __init__(self, dataset, patch_image_size=(512, 512), patch_steps=(128, 128), transform_retained_for_original_data=None, transform_only_for_original_data=None):

        print('PatchSplit Dataset created')

        self.dataset = dataset

        self.transform = dataset.transform
        self.rng = dataset.rng
        self.transform_only_valid_centers = dataset.transform_only_valid_centers
        self.transform_per_sample_rng = dataset.transform_per_sample_rng

        dataset.transform = None

        # retain ToTensor transform to ensure output from underlying dataset is in Tensors
        if transform_retained_for_original_data is None:
            transform_retained_for_original_data = ['ToTensor']

        if self.transform is not None:
            if transform_only_for_original_data is not None:
                transform_retained_for_original_data += transform_only_for_original_data

            if type(self.transform) == list:
                dataset.transform = [t for t in self.transform if t['name'] in transform_retained_for_original_data]
                if transform_only_for_original_data is not None:
                    self.transform = [t for t in self.transform if t['name'] not in transform_only_for_original_data]                    
            else:
                dataset.transform = my_transforms.Compose([t for t in self.transform.transforms
                                                                if any([isinstance(t,getattr(globals()['my_transforms'], s)) for s in transform_retained_for_original_data])])

                if transform_only_for_original_data is not None:
                    self.transform = my_transforms.Compose([t for t in self.transform.transforms
                                                                if all([not isinstance(t,getattr(globals()['my_transforms'], s)) for s in transform_only_for_original_data])])

        self.last_n = -1
        self.last_image_and_gt = None

        self._register_image_patches(dataset, patch_image_size, patch_steps)

    def _register_image_patches(self, dataset, output_image_size, steps):

        # filenames of original images
        self.image_list = []

        self.output_image_size = output_image_size
        self.patch_mapping = {}
        self.org_img_size_mapping = {}
        index = 0
        print('Registering image patches ')

        trsf = my_transforms.get_transform(dataset.transform ) if dataset.transform is not None and type(dataset.transform ) == list else dataset.transform 
        dataset_return_cfg = (dataset.return_image, dataset.transform)
        dataset.return_image = False
        dataset.transform = None

        # calculate adjustment func for size based on transform
        def size_adjustmet_from_transform(imsize,T):
            if trsf is not None:
                for t in trsf.transforms:
                    if isinstance(t,my_transforms.Padding):
                        imsize = imsize[0] + t.borders[0] + t.borders[2], imsize[1] + t.borders[1] + t.borders[3]
                    elif isinstance(t, my_transforms.Resize):
                        imsize = t.size
                    elif type(t) in [getattr(globals()['my_transforms'],N)
                                        for N in ['Normalize','RandomGaussianBlur', 'ToTensor', 'ColorJitter',
                                                  'RandomVerticalFlip', 'RandomHorizontalFlip']]:
                        pass
                    else:
                        raise Exception("Unsupported transform for use in PatchSplitDataset")
            return imsize
        for n, sample in enumerate(tqdm(dataset)):
            self.image_list.append(sample['im_name'])
            if 'im_size' in sample:
                imsize = sample['im_size']
            else:
                assert 'image' in sample
                im = sample['image']
                if isinstance(im,torch.Tensor):
                    imsize = im.shape[-2:][::-1]
                elif isinstance(im,np.ndarray):
                    imsize = im.shape[:2][::-1]
                elif isinstance(im,Image.Image):
                    imsize = im.size
                else:
                    raise Exception("Unsupporeted input image data format")

            imsize = size_adjustmet_from_transform(imsize, trsf)

            new_patches_list = [(n, i, j) for j in np.arange(0, imsize[1] - output_image_size[1] + steps[1], steps[1])
                                for i in np.arange(0, imsize[0] - output_image_size[0] + steps[0], steps[0])]

            self.patch_mapping.update({index + n: p for n, p in enumerate(new_patches_list)})
            self.org_img_size_mapping.update({index + n: imsize for n, _ in enumerate(new_patches_list)})

            index += len(new_patches_list)

        # restore dataset
        dataset.return_image, dataset.transform = dataset_return_cfg

        self.real_size = len(self.image_list)
        self.num_patches = len(self.patch_mapping)
        print(' done\n number of images: %d\n number of patches: %d' % (self.real_size, self.num_patches))

    def _get_image_patch_index(self, index):
        n, x, y = self.patch_mapping[index]
        w, h = self.output_image_size

        roi_x = np.arange(x, x + w)
        roi_y = np.arange(y, y + h)

        org_w, org_h = self.org_img_size_mapping[index]

        return n, x, y, w, h, roi_x, roi_y, org_w, org_h

    def __len__(self):
        return self.num_patches

    def __getitem__(self, index):
        if index < 0 or index >= self.num_patches:
            raise IndexError()

        n, x, y, w, h, roi_x, roi_y, org_w, org_h = self._get_image_patch_index(index)

        sample = {}

        filename = os.path.basename(self.image_list[n])
        folder = os.path.dirname(self.image_list[n])
        
        filename_base, filename_ext = os.path.splitext(filename)
        patch_name = os.path.join(folder, f'{filename_base}_patch{w}x{h}_i={x}_j={y}{filename_ext}')

        # load image and decode instance/label masks
        if self.last_n == n:
            image_sample = self.last_image_and_gt
        else:
            image_sample = self.dataset[n]

            self.last_n = n
            self.last_image_and_gt = image_sample

        image = image_sample['image']
        center = image_sample['center']

        # clamp roi to max size
        roi_x = roi_x[roi_x < image.shape[-1]]
        roi_y = roi_y[roi_y < image.shape[-2]]

        assert isinstance(image, torch.Tensor) and len(image.shape) == 3
        assert (isinstance(center, torch.Tensor) or isinstance(center, np.ndarray)) and len(center.shape) == 2 and center.shape[1] == 2

        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center)

        # center need to be cloned to allow for in-place changes while re-using original data later on
        center = center.clone()

        # crop to ROI patch only
        image = image[:,roi_y][:,:, roi_x]

        # zero-out all centers outside of ROI
        valid_ids = np.where((center[:, 0] >= x) & (center[:, 0] < x + w) &
                                (center[:, 1] >= y) & (center[:, 1] < y + h))

        # offset center values to ROI position
        center[valid_ids] = center[valid_ids] - torch.as_tensor(np.array([x, y]))

        # set all remaining values to zero
        invalid_mask = torch.ones(len(center),dtype=torch.bool)
        invalid_mask[valid_ids] = 0
        center[invalid_mask, :] = 0


        # pad the width and height if needed
        padding = [self.output_image_size[0] - image.shape[-1],
                   self.output_image_size[1] - image.shape[-2]]
        if padding[0] > 0 or padding[1] > 0:
            torch_pad = lambda I: F.pad(I, (0, 0, padding[0], padding[1]), 0, 'constant')
            numpy_pad = lambda I: np.pad(I, [(0, padding[1]), (0, padding[0])] + [(0,0)] * (len(I.shape) - 2), constant_values=0, mode='constant')
            image, = [numpy_pad(I) if isinstance(I,np.ndarray) else torch_pad(I) for I in [image,]]

        sample['image'] = image
        sample['im_name'] = patch_name
        sample['index'] = index

        sample['center'] = center
        sample['grid_index'] = np.array([n, x, y, w, h, org_w, org_h])

        # transform
        transform = my_transforms.get_transform(self.transform) if self.transform is not None and type(self.transform) == list else self.transform
        if transform is not None:

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
                        if len(out_of_bounds_ids) < len(center): # at least one must be present
                            do_transform = False
                            sample = new_sample
                    elif type(self.transform_only_valid_centers) == int:
                        if len(center) - len(out_of_bounds_ids) >= self.transform_only_valid_centers:
                            do_transform = False
                            sample = new_sample
                    elif type(self.transform_only_valid_centers) == float:
                        min_visible = int(self.transform_only_valid_centers * len(center))
                        if len(center) - len(out_of_bounds_ids) >= min_visible:
                            do_transform = False
                            sample = new_sample
                    else:
                        raise Exception("Invalid type of transform_only_valid_centers, allowed types: bool, int, float")                
                ii += 1

        return sample

if __name__ == "__main__":
    from utils import transforms as my_transforms

    transform = [
        {
            'name': 'ToTensor',
            'opts': {
                'keys': ('image',),
                'type': (torch.FloatTensor, ),
            }
        },
    ]

    from datasets.ViCoSClothDataset import ClothDataset

    subfolders = [dict(folder='bg=white_desk', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
        dict(folder='bg=green_checkered', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
        dict(folder='bg=poster', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
        dict(folder='bg=festive_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
        ]

    normals_mode = 2
    use_normals = False

    db = ClothDataset(root_dir='/storage/datasets/ClothDataset/ClothDatasetVICOS/', resize_factor=0.5, transform_only_valid_centers=1.0, transform=transform, subfolders=subfolders, remove_out_of_bounds_centers=False,
                      #valid_img_names=['bg=white_desk/cloth=big_towel/rgb/image_0000_view0_']
                      )
    patch_db = PatchSplitDataset(db, patch_image_size=(512, 512), patch_steps=(448,128))
    for i,item in enumerate(tqdm(patch_db)):
        #if item['index'] % 100 == 0:
        #    print('loaded index %d' % item['index'])
        if True:
            center = item['center']
            gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]

            plt.clf() 
            plt.subplot(1, 2, 1)
            plt.imshow(item['image'].permute([1, 2, 0]))
            plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')

            plt.draw(); plt.pause(0.01)
            plt.waitforbuttonpress()
