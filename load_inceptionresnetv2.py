# -*- coding: utf-8 -*-
"""
Created on Fri May 31 00:44:01 2019

@author: Max
"""

import pickle
from inceptionresnetv2 import InceptionResNetV2

def load_inceptionresnetv2(balanced_dir='D:/steep_training/ski-race/balanced/'):
    
    balanced_one_hot_filename = balanced_dir + 'one_hot_dict.pkl'
    
    with open(balanced_one_hot_filename, 'rb') as handle:
        balanced_one_hot = pickle.load(handle)
        
    num_classes = len(balanced_one_hot)
    
    model = InceptionResNetV2(num_classes=num_classes)
    
    return model