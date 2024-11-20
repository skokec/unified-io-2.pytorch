import glob
import os, cv2

import numpy as np
from PIL import Image

import json,  os

import pylab as plt

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

    def __init__(self, preprocessor, dataset, targets_definition=None, returned_raw_sample=False, randomize_keypoints_order=True, jitter_keypoints_px=False, apply_internal_scale_aug=False, PLOT=False):
        self.preprocessor = preprocessor
        self.dataset = dataset

        self.returned_raw_sample = returned_raw_sample

        self.prompt = Prompt()

        self.randomize_keypoints_order = randomize_keypoints_order
        self.jitter_keypoints_px = jitter_keypoints_px
        self.apply_internal_scale_aug = apply_internal_scale_aug

        if targets_definition is None:
            # set to use keypoints by default
            targets_definition = [dict(type='keypoints', prob=1.0,
                                    groundruth_key='center',
                                    prompt_key='Towel_Corners')]

        self.targets_definition = targets_definition
        self.targets_types, self.targets_prob = zip(*[(i, t['prob']) for i,t in enumerate(self.targets_definition)])

        self.PLOT = PLOT

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        sample = self.dataset[index]

        return self.preprocess_sample(sample)
    
    def preprocess_sample(self, sample):
        
        img = sample['image']  
        # prepare args for preprocessor
        preprocessor_args = dict(image_inputs=np.transpose(img,(1,2,0)),
                                 is_training=self.apply_internal_scale_aug,)


        # we can only process one target per sample, randomly chose one based on probabilities
        selected_target_id = np.random.choice(self.targets_types, p=self.targets_prob)


        selected_target = self.targets_definition[selected_target_id]
   
        if selected_target['type'].lower() == 'keypoints':
            gt_key = selected_target['groundruth_key']
            prompt_key = selected_target['prompt_key']


            # set text target and set function to generate text from keypoints
            from functools import partial
            preprocessor_args.update(dict(text_targets='', 
                                          target_modality='text',
                                          raw_features_fn=partial(self._generate_keypoint_target, keypoints=sample[gt_key])))
                                         

        elif selected_target['type'].lower() == 'mask':
            gt_key = selected_target['groundruth_key'] 
            prompt_key = selected_target['prompt_key'] 

            # set image target and set function to generate text from keypoints
            preprocessor_args.update(dict(image_targets=sample[gt_key], 
                                          target_modality='image'))
        else:
            raise Exception("Invalid target type for KeypointPreprocessorDataset (only 'keypoints' or 'mask' allowed)")

        # generate appropriate input prompt
        input = self.prompt.random_prompt(prompt_key)
        
        # do preprocessing
        preprocessed_example = self.preprocessor(text_inputs=input, **preprocessor_args)
        
        if 'index' in sample:
            preprocessed_example['index'] = sample['index']
        if 'im_size' in sample:
            preprocessed_example['im_size'] = sample['im_size']
        if 'im_name' in sample:
            preprocessed_example['im_name'] = sample['im_name']
        
        if self.returned_raw_sample:
            preprocessed_example['sample'] = sample

        return preprocessed_example

    # use this function to create keypoints based on actual image translation and convert them to text_target
    def _generate_keypoint_target(self, features, keypoints):
        from uio2 import config            
        gt_centers = keypoints[(keypoints[:, 0] > 0) | (keypoints[:, 1] > 0), :]
        if len(gt_centers) == 0:
            features['text_targets'] = "No points found"
            return features
        # convert them to transformed input image space
        top_pad, left_pad, scale, in_height, in_width,_,_,off_y,off_x,_,_ = np.array(features['meta/image_info'])
        # add scale            
        gt_centers /= scale

        # add padding
        gt_centers[:,0] += left_pad - off_x
        gt_centers[:,1] += top_pad - off_y

        # swicth x,y 
        gt_centers = gt_centers[:,[1,0]]

        # remove keypoints that could fall outside of image (e.g., due to internal scale augmentation that also crops the image)
        gt_centers = np.array([[x,y] for x,y in gt_centers if x >=0 and x<config.IMAGE_INPUT_SIZE[1]-1 and y >=0 and y<config.IMAGE_INPUT_SIZE[0]-1])

        if len(gt_centers) == 0:
            features['text_targets'] = "No points found"
            return features

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


        gt_centers_text = centers_to_tokens(gt_centers, config.IMAGE_INPUT_SIZE)
        features['text_targets'] = gt_centers_text
        
        if self.PLOT:
            img = features["image_inputs"]
            plt.clf()
            plt.subplot(1, 1, 1)            
            #plt.imshow(np.transpose(img,(1,2,0)))
            plt.imshow(img)
            
            plt.draw(); plt.pause(0.01)
            plt.waitforbuttonpress()	

        return features
        


