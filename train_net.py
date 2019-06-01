# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:51:50 2019

@author: Max
"""
from inceptionresnetv2 import get_inceptionresnetv2
from data_loader import GameFrameData
from test_train_split import get_test_train_partition
import torch.optim as optim
from torch.utils import data

import pickle

label_dict_filename = 'D:/steep_training/ski-race/balanced/label_dict.pkl'
with (label_dict_filename, 'rb') as handle:
    label_dict = pickle.load(handle)

device = torch.device('cuda:0')
cudnn.benchmark = True

params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 6}

partition = get_test_train_partition(label_dict)

training_set = GameFrameData(partition['train'], label_dict)
validation_set = GameFrameData(partition['validation'], label_dict)

training_generator = data.DataLoader(training_set, **params)
validation_generator = data.DataLoader(validation_set, **params)

initial_epoch = 0
max_epoch = 30

for epoch in range(initial_epoch, max_epoch):
    # Train
    for local_batch, local_labels in training_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        
    # Validation
    for local_batch, local_labels in validation_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)   

    

model = get_inceptionresnetv2()


import torch.optim as optim

optimizer = optim.Adam(net.params(), lr=0.001)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

for epoch in range(start_epoch, n_epoch):
    
    