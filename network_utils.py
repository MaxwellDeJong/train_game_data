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

def load_label_dict():

    label_dict_filename = 'D:/steep_training/ski-race/balanced/label_dict.pkl'
    with open(label_dict_filename, 'rb') as handle:
        label_dict = pickle.load(handle)
        
    return label_dict

def load_normalization_stats():
    
    stats_arr = np.load('D:/steep_training/ski-race/balanced/normalization_weights.npy')
    
    means = stats_arr[0]
    stds = stats_arr[1]
    
    return (means, stds)


def load_generators(params):
    
    label_dict = load_label_dict()
    (means, stds) = load_normalization_stats()
    
    frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
            ])

    partition = get_test_train_partition(label_dict)

    training_set = GameFrameData(partition['train'], label_dict, frame_transform)
    validation_set = GameFrameData(partition['validation'], label_dict, frame_transform)

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    
    return (training_generator, validation_generator)