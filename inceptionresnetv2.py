# -*- coding: utf-8 -*-
"""
Created on Fri May 31 00:44:01 2019

@author: Max
"""

import importlib.util
import pickle

spec = importlib.util.spec_from_file_location('module.name', 
    'C:/Users/Max/Documents/pretrained-models.pytorch/pretrainedmodels/models/inceptionresnetv2.py')
inception = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inception)

def get_inceptionresnetv2(balanced_dir='D:/steep_training/ski-race/balanced/'):
    
    balanced_one_hot_filename = balanced_dir + 'one_hot_dict.pkl'
    
    with open(balanced_one_hot_filename, 'rb') as handle:
        balanced_one_hot = pickle.load(handle)
        
    num_classes = len(balanced_one_hot)
    
    model = inception.get_inceptionresnetv2(num_classes=num_classes, pretrained=None)
    
    return model