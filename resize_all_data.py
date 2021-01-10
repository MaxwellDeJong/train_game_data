# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 23:55:04 2019

@author: Max
"""

import cv2
import numpy as np
import os

def save_resized_img(img_idx, validation=False, loc='E:/full_size/', compress=True):
    
    if validation:
        frame_label = 'validation'
    else:
        frame_label = 'training'
        
    filename = frame_label + '_frame-' + str(img_idx) + '.npy'
    original_filename = 'F:/balanced/' + filename
    new_filename = loc + filename
    
    valid_file = os.path.isfile(original_filename)
    
    if valid_file:
    
        sample = np.load(original_filename)
        sample_resized = cv2.imdecode(sample, 1)
        #sample_decompressed = cv2.imdecode(sample, 1)
        
        #sample_resized = cv2.resize(sample_decompressed, (224, 224))
        
        if compress:
            (result, comp_resized) = cv2.imencode('.jpg', sample_resized)
        
            if result:
                np.save(new_filename, comp_resized)
            
            else:
                print('Error. Invalid image compression.')
                
        else:
            np.save(new_filename, sample_resized)
            
    return valid_file
            

def resize_all_imgs():
    
    for validation_frame in [True, False]:
    
        valid_img = True
        img_idx = 0
        
        while valid_img:
            valid_img = save_resized_img(img_idx, validation_frame, compress=False)
            img_idx += 1
            
    print('All images resized.')
    
    
resize_all_imgs()