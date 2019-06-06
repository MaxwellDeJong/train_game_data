# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:44:31 2019

@author: Max
"""


def get_test_train_partition(training_label_dict, validation_label_dict):
    
    n_training_frames = len(training_label_dict)
    training_id_list = range(n_training_frames)

    n_validation_frames = len(validation_label_dict)
    validation_id_list = range(n_validation_frames)
    
    partition = {'train': training_id_list, 'validation': validation_id_list}
    
    return partition
