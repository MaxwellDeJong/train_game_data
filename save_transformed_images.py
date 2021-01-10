# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:24:25 2019

@author: Max
"""

from network_utils import load_normalization_stats
from torchvision import transforms
import numpy as np
import os
import torch

def get_transform(dirname='ski-race'):

    (means, stds) = load_normalization_stats(dirname=dirname)
    
    frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
            ])
    
    return frame_transform


def save_transformed_img(img_idx, validation=False, dirname='ski-race'):
    
    original_loc = 'D:/steep_training/' + dirname + '/balanced/'
    new_loc = 'F:/' + dirname + '/'
    
    if validation:
        frame_label = 'validation'
    else:
        frame_label = 'training'
        
    frame_transform = get_transform(dirname=dirname)
        
    original_filename = original_loc + frame_label + '_frame-' + str(img_idx) + '.npy'
    new_filename = new_loc + frame_label + '_frame-' + str(img_idx) + '.pt'
    
    valid_file = os.path.isfile(original_filename)
    
    if valid_file:
    
        sample = np.load(original_filename)
        sample_transformed = frame_transform(sample)
        
        torch.save(sample_transformed, new_filename)
            
    return valid_file
            

def transform_all_imgs(dirname='ski-race'):
    
    for validation_frame in [True, False]:
        if validation_frame:
            print('Starting validation frames...')
        else:
            print('Starting test frames...')
    
        valid_img = True
        img_idx = 0
        
        while valid_img:
            valid_img = save_transformed_img(img_idx, validation=validation_frame, dirname=dirname)
            img_idx += 1
            
            if (img_idx % 5000 == 0):
                print('Finished transforming', img_idx, 'frames')
            
    print('All images transformed.')


transform_all_imgs(dirname='wing-suit')