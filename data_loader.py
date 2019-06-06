# -*- coding: utf-8 -*-
"""
Inspired by the tutorial by Afshine Amidi and Shervine Amidi
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import torch
import numpy as np
from torch.utils import data
from cv2 import imdecode
    

def npy_loader(filename, transform):
    
    sample = np.load(filename)
    sample_decompressed = imdecode(sample, 1)
    
    return transform(sample_decompressed)


class GameFrameData(data.Dataset):
    
    def __init__(self, list_IDs, labels, transform, train=True):
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

        if train:
            self.file_prefix = 'training'
        else:
            self.file_prefix = 'validation'
        
    def __len__(self):
        
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
        
        ID = self.list_IDs[idx]

        filename = 'D:/steep_training/ski-race/balanced/' + self.file_prefix + '_frame-{}.npy'.format(ID)

        X = npy_loader(filename, self.transform)
        y = torch.LongTensor(self.labels[ID])
        
        return (X, y)
