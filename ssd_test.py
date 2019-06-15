# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 11:13:42 2019

@author: Max
"""
import numpy as np
import pickle
from data_loader import GameFrameData
from torch.utils import data
from torchvision import transforms

def get_test_train_partition(training_label_dict):
    
    n_training_frames = len(training_label_dict)
    training_id_list = range(n_training_frames)
    
    partition = {'train': training_id_list, 'validation': training_id_list[:]}
    
    return partition


def load_label_dict():
    
    training_label_filename = 'D:/steep_training/ski-race/balanced/training_label_dict.pkl'

    with open(training_label_filename, 'rb') as handle:
        training_label_dict = pickle.load(handle)
        
    sm_dict = {}
    
    for key in range(20224):
        sm_dict[key] = training_label_dict[key]
        
    return sm_dict


def load_normalization_stats():
    
    stats_arr = np.load('C:/Users/Max/Documents/balanced/normalization_weights.npy')
    
    means = stats_arr[0]
    stds = stats_arr[1]
    
    return (means, stds)


def load_generators(params):
    
    training_label_dict = load_label_dict()
    (means, stds) = load_normalization_stats()
    
    frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
            ])

    partition = get_test_train_partition(training_label_dict)

    training_set = GameFrameData(partition['train'], training_label_dict, frame_transform)
    validation_set = GameFrameData(partition['validation'], training_label_dict, frame_transform)
    
    print('training set: ', training_set)

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    
    return (training_generator, validation_generator)
