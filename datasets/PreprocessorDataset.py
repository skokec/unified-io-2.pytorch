import glob
import os, cv2

import numpy as np
from PIL import Image

import json,  os

import torch
from torch.utils.data import Dataset

from uio2.data_utils import values_to_tokens

def centers_to_tokens(gt_centers, img_shape):
    
    gt_centers_tokens_x = values_to_tokens(gt_centers[:,0] / img_shape[1])
    gt_centers_tokens_y = values_to_tokens(gt_centers[:,1] / img_shape[0])


    gt_centers_tokens_x = [x.decode("utf-8") for x in gt_centers_tokens_x.numpy()]
    gt_centers_tokens_y = [y.decode("utf-8") for y in gt_centers_tokens_y.numpy()]

    gt_centers_text = " , ".join([f"{x} {y}" for x,y in zip(gt_centers_tokens_x, gt_centers_tokens_y)])

    return gt_centers_text

class PreprocessorDataset(Dataset):

    def __init__(self, preprocessor, dataset, returned_raw_sample=False):
        self.preprocessor = preprocessor
        self.dataset = dataset

        self.returned_raw_sample = returned_raw_sample

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        sample = self.dataset[index]

        return self.preprocess_sample(sample)
    
    def preprocess_sample(self, sample):
        img = sample['image']

        # get gt points
        center = sample['center']
        
        def translate_gt(features, center):
            gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]
            # convert them to transformed input image space
            top_pad, left_pad, scale, in_height, in_width,_,_,_,_,off_y,off_x = np.array(features['meta/image_info'])
            # add scale            
            gt_centers /= scale

            # add padding
            gt_centers[:,0] += left_pad
            gt_centers[:,1] += top_pad

            # swicth x,y 
            gt_centers = gt_centers[:,[1,0]]

            from uio2 import config
            gt_centers_text = centers_to_tokens(gt_centers, config.IMAGE_INPUT_SIZE)
            features['text_targets'] = gt_centers_text

            #print(sample['im_name'])
            #print("gt:", gt_centers_text)
            #print(gt_centers)

            return features

        from functools import partial
        imput = "List coordinates of all visible towel corners in <image_input>"
        preprocessed_example = self.preprocessor(text_inputs=imput, image_inputs=np.transpose(img,(1,2,0)), text_targets="", target_modality="text", 
                                                 raw_features_fn=partial(translate_gt, center=center))
        
        if 'index' in sample:
            preprocessed_example['index'] = sample['index']
        if 'im_size' in sample:
            preprocessed_example['im_size'] = sample['im_size']
        if 'im_name' in sample:
            preprocessed_example['im_name'] = sample['im_name']
        
        if self.returned_raw_sample:
            preprocessed_example['sample'] = sample

        return preprocessed_example
        


