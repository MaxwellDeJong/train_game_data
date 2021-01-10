# -*- coding: utf-8 -*-
"""
Inspired by the tutorial by Afshine Amidi and Shervine Amidi
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import torch
import numpy as np
from torch.utils import data
from cv2 import imdecode


class GameFrameData(data.Dataset):
    
    def __init__(self, list_IDs, labels, train=True, dirname='ski-race'):
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.dirname= dirname

        if train:
            self.file_prefix = 'training'
        else:
            self.file_prefix = 'validation'
        
    def __len__(self):
        
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
        
        ID = self.list_IDs[idx]

        filename = 'F:/' + self.dirname + '/' + self.file_prefix + '_frame-{}.pt'.format(ID)

        X = torch.load(filename)
        y = torch.LongTensor(self.labels[ID])
        
        return (X, y)
