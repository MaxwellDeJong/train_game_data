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


def load_label_dicts():

    training_label_filename = 'D:/steep_training/ski-race/balanced/training_label_dict.pkl'
    validation_label_filename = 'D:/steep_training/ski-race/balanced/validation_label_dict.pkl'

    with open(training_label_filename, 'rb') as handle:
        training_label_dict = pickle.load(handle)

    with open(validation_label_filename, 'rb') as handle:
        validation_label_dict = pickle.load(handle)
        
    return (training_label_dict, validation_label_dict)


def load_normalization_stats():
    
    stats_arr = np.load('D:/steep_training/ski-race/balanced/normalization_weights.npy')
    
    means = stats_arr[0]
    stds = stats_arr[1]
    
    return (means, stds)


def load_generators(params):
    
    (training_label_dict, validation_label_dict) = load_label_dict()
    (means, stds) = load_normalization_stats()
    
    frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
            ])

    partition = get_test_train_partition(label_dict)

    training_set = GameFrameData(partition['train'], training_label_dict, frame_transform)
    validation_set = GameFrameData(partition['validation'], validation_label_dict, frame_transform)

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    
    return (training_generator, validation_generator)
