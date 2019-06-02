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
    
#    sample_torch = torch.from_numpy(sample_decompressed)

    return transform(sample_decompressed)

class GameFrameData(data.Dataset):
    
    def __init__(self, list_IDs, labels, transform):
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        
    def __len__(self):
        
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
        
        ID = self.list_IDs[idx]
        
        X = npy_loader('D:/steep_training/ski-race/balanced/training_frame-{}.npy'.format(ID), self.transform)
        
        y = torch.LongTensor(self.labels[ID])
        
        return (X, y)