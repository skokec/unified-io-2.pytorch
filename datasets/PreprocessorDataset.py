import glob
import os, cv2

import numpy as np
from PIL import Image

import json,  os

import torch
from torch.utils.data import Dataset

from uio2.data_utils import values_to_tokens

from uio2.prompt import Prompt

def centers_to_tokens(gt_centers, img_shape):
    
    gt_centers_tokens_x = values_to_tokens(gt_centers[:,0] / img_shape[1])
    gt_centers_tokens_y = values_to_tokens(gt_centers[:,1] / img_shape[0])


    gt_centers_tokens_x = [x.decode("utf-8") for x in gt_centers_tokens_x.numpy()]
    gt_centers_tokens_y = [y.decode("utf-8") for y in gt_centers_tokens_y.numpy()]

    gt_centers_text = " , ".join([f"{x} {y}" for x,y in zip(gt_centers_tokens_x, gt_centers_tokens_y)])

    return gt_centers_text

class KeypointPreprocessorDataset(Dataset):

    def __init__(self, preprocessor, dataset, returned_raw_sample=False, randomize_keypoints_order=True, jitter_keypoints_px=False, apply_internal_scale_aug=False):
        self.preprocessor = preprocessor
        self.dataset = dataset

        self.returned_raw_sample = returned_raw_sample

        self.prompt = Prompt()

        self.randomize_keypoints_order = randomize_keypoints_order
        self.jitter_keypoints_px = jitter_keypoints_px
        self.apply_internal_scale_aug = apply_internal_scale_aug

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        sample = self.dataset[index]

        return self.preprocess_sample(sample)
    
    def preprocess_sample(self, sample):
        img = sample['image']

        # get gt points
        center = sample['center']
        
        # use this function to create keypoints based on actual image translation and convert them to text_target
        def translate_gt(features, center):
            gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]
            if len(gt_centers) == 0:
                features['text_targets'] = "No points found"
                return features
            # convert them to transformed input image space
            top_pad, left_pad, scale, in_height, in_width,_,_,_,_,off_y,off_x = np.array(features['meta/image_info'])
            # add scale            
            gt_centers /= scale

            # add padding
            gt_centers[:,0] += left_pad
            gt_centers[:,1] += top_pad

            # swicth x,y 
            gt_centers = gt_centers[:,[1,0]]

            if self.randomize_keypoints_order:
                # Randomly shuffle the order of keypoints
                indices = np.random.permutation(gt_centers.shape[0])
                gt_centers = gt_centers[indices]

            if self.jitter_keypoints_px:
                # jitter each keypoint to prevent overfitting to the exact values

                # Generate random jitter for each coordinate (N, 2)
                jitter = np.random.uniform(-self.jitter_keypoints_px, self.jitter_keypoints_px, size=gt_centers.shape)

                # Add the jitter to the original centers
                gt_centers += jitter

                # clip it within range
                gt_centers[:,0] = np.clip(gt_centers[:,0],0, config.IMAGE_INPUT_SIZE[1]-1)
                gt_centers[:,1] = np.clip(gt_centers[:,1],0, config.IMAGE_INPUT_SIZE[0]-1)


            from uio2 import config
            gt_centers_text = centers_to_tokens(gt_centers, config.IMAGE_INPUT_SIZE)
            features['text_targets'] = gt_centers_text

            return features

        from functools import partial
        
        input = self.prompt.random_prompt('Towel_Corners')

        preprocessed_example = self.preprocessor(text_inputs=input, image_inputs=np.transpose(img,(1,2,0)), text_targets="", target_modality="text", 
                                                 is_training=self.apply_internal_scale_aug, raw_features_fn=partial(translate_gt, center=center),)
        
        if 'index' in sample:
            preprocessed_example['index'] = sample['index']
        if 'im_size' in sample:
            preprocessed_example['im_size'] = sample['im_size']
        if 'im_name' in sample:
            preprocessed_example['im_name'] = sample['im_name']
        
        if self.returned_raw_sample:
            preprocessed_example['sample'] = sample

        return preprocessed_example
        


