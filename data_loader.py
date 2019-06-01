# -*- coding: utf-8 -*-
"""
Inspired by the tutorial by Afshine Amidi and Shervine Amidi
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import torch
from torch.utils import data
from cv2 import imdecode

class GameFrameData(data.Dataset):
    
    def __init__(self, list_IDs, labels):
        
        self.labels = labels
        self.list_IDs = list_IDs
        
    def __len__(self):
        
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
        
        ID = self.list_IDs[idx]
        
        raw_X = torch.load('D:/steep_training/ski-race/balanced/training_frame-{}.npy'.format(ID))
        
        X = imdecode(raw_X, 1)
        y = self.labels[ID]
        
        return (X, y)