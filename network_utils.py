# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:32:39 2019

@author: Max
"""
import numpy as np
import pickle
from data_loader import GameFrameData
from test_train_split import get_test_train_partition
from torch.utils import data
from torchvision import transforms


def load_label_dicts(dirname='ski-race'):

    training_label_filename = 'E:/steep_training/' + dirname + '/balanced/training_label_dict.pkl'
    validation_label_filename = 'E:/steep_training/' + dirname + '/balanced/validation_label_dict.pkl'

    with open(training_label_filename, 'rb') as handle:
        training_label_dict = pickle.load(handle)

    with open(validation_label_filename, 'rb') as handle:
        validation_label_dict = pickle.load(handle)
        
    return (training_label_dict, validation_label_dict)


def load_normalization_stats(dirname='ski-race'):
    
    stats_arr = np.load('E:/steep_training/' + dirname + '/balanced/normalization_weights.npy')
    
    means = stats_arr[0]
    stds = stats_arr[1]
    
    return (means, stds)


def load_generators(params, dirname='ski-race'):
    
    (training_label_dict, validation_label_dict) = load_label_dicts(dirname=dirname)

    partition = get_test_train_partition(training_label_dict, validation_label_dict)

    training_set = GameFrameData(partition['train'], training_label_dict, dirname=dirname)
    validation_set = GameFrameData(partition['validation'], validation_label_dict, dirname=dirname)

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    
    return (training_generator, validation_generator)
