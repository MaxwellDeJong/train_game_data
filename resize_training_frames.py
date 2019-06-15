# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:48:41 2019

@author: Max
"""

import numpy as np
import cv2
import os

def cvtFrame(idx, training=True):
    
    desired_size = (320, 180)
    
    if training:
        file_prefix = 'training'
    else:
        file_prefix = 'validation'
        
    filename = 'D:/steep_training/ski-race/balanced/' + file_prefix + '_frame-{}.npy'.format(idx)
    new_filename = 'F:/balanced/' + file_prefix + '_frame-{}.npy'.format(idx)
    
    if (os.path.isfile(filename)):
        img_comp = np.load(filename)
        img = cv2.imdecode(img_comp, 1)
        
        new_img = cv2.resize(img, desired_size)
        
#        cv2.imshow('Small image', new_img)
#        cv2.waitKey()
        
        (_, new_img_comp) = cv2.imencode('.jpg', new_img)

        np.save(new_filename, new_img_comp)
        
        return True
    
    return False

def cvt_all():
    
    training = True
    valid_file = True
    
    idx = 0
    
    while valid_file:
        valid_file = cvtFrame(idx, training)
        idx += 1
        
    training = False
    valid_file = True
    
    idx = 0
    
    while valid_file:
        valid_file = cvtFrame(idx, training)
        idx += 1
        
cvt_all()