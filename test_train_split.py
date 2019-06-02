# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:44:31 2019

@author: Max
"""
from sklearn.model_selection import train_test_split

def get_test_train_partition(label_dict):
    
    n_frames = len(label_dict)
    id_list = range(n_frames)
    
    X_train, X_valid, _, _ = train_test_split(id_list, id_list, test_size=0.1)
    
    partition = {'train': X_train, 'validation': X_valid}
    
    return partition