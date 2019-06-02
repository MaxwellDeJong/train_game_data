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

def main():

    device = torch.device('cuda:0')
    cudnn.benchmark = True
    
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 6}
    
    (training_generator, validation_generator) = load_generators(params)
    
    model = load_inceptionresnetv2()
    #writer = SummaryWriter('D:/steep_training/ski-race/balanced/log/')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model = model.to(device)
    
    initial_epoch = 0
    max_epoch = 30
    
    curr_batch = 0
    
    for epoch in range(initial_epoch, max_epoch):
        # Train
        for local_batch, local_labels in training_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(local_batch)
            loss = criterion(logps, torch.max(local_labels, 1)[1])
            loss.backward()
            optimizer.step()
            
            if (curr_batch % 16 == 0):
                print('Have analyzed ', curr_batch * 16, ' frames')              
                            
            curr_batch += 1
            
        # Validation
        with torch.set_grad_enabled(False):
            
            cum_loss = 0
            accuracy = 0
            
            for local_batch, local_labels in validation_generator:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                logps = model.forward(local_batch)
                loss = criterion(logps, torch.max(local_labels, 1)[1])
                cum_loss += loss
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(dim=1)
                equals = (top_class == local_labels(*top_class.shape))
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            print('FINISHED EPOCH ', epoch)
            print('Validation loss: ', cum_loss / len(validation_generator))
            print('Validation accuracy: ', accuracy / len(validation_generator))
            
        torch.save(model, 'D:/steep_training/ski-race/balanced/model/inceptionresnetv2.pth')
        
if __name__ == '__main__':
    main()