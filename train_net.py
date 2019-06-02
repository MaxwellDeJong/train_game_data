# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:51:50 2019

@author: Max
"""
from load_inceptionresnetv2 import load_inceptionresnetv2
from network_utils import load_generators
import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

def main(load_model):

    device = torch.device('cuda:0')
    cudnn.benchmark = True
    
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 6}
    
    (training_generator, validation_generator) = load_generators(params)
    
    if load_model:
        model = torch.load('D:/steep_training/ski-race/balanced/model/inceptionresnetv2.pth')
    else:
        model = load_inceptionresnetv2()
    #writer = SummaryWriter('D:/steep_training/ski-race/balanced/log/')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    model = model.to(device)
    
    initial_epoch = 30
    max_epoch = 40
    
    for epoch in range(initial_epoch, max_epoch):
        
        curr_batch = 0
        running_loss = 0.0
        # Train
        for local_batch, local_labels in training_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(local_batch)
            loss = criterion(logps, torch.max(local_labels, 1)[1])
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (curr_batch % 16 == 0):
                print('Have analyzed ', curr_batch * 16, ' frames in epoch ', epoch)
                
                if (curr_batch % 80 == 0):
        
                    print('Batch loss: ', running_loss / 80)
                    running_loss = 0.0
                            
            curr_batch += 1
            
        # Validation
        with torch.set_grad_enabled(False):
            
            cum_loss = 0
            accuracy = 0
            
            for local_batch, local_labels in validation_generator:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                logps = model.forward(local_batch)
                loss = criterion(logps, torch.max(local_labels, 1)[1])
                cum_loss += loss.item()
                
                ps = torch.exp(logps)
                max_preds = torch.max(ps, 1)[1]
                max_true = torch.max(local_labels, 1)[1]
                accuracy += max_preds.eq(max_true).sum().item()
                
            print('FINISHED EPOCH ', epoch)
            print('Validation loss: ', cum_loss / len(validation_generator))
            print('Validation accuracy: ', accuracy / len(validation_generator))
            
        torch.save(model, 'D:/steep_training/ski-race/balanced/model/inceptionresnetv2.pth')
        
if __name__ == '__main__':
    main(True)