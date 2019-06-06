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


def save_checkpoint(state):

    filename = 'D:/steep_training/ski-race/balanced/model/inceptionresnetv2.pth'

    torch.save(state, filename)


def load_checkpoint(model, optimizer):

    filename = 'D:/steep_training/ski-race/balanced/model/inceptionresnetv2.pth'

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']


def main(load_model):

    device = torch.device('cuda:0')
    cudnn.benchmark = True
    
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 6}
    
    (training_generator, validation_generator) = load_generators(params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    model = load_inceptionresnetv2()

    if (load_model):
        initial_epoch = load_checkpoint(model)

    else:
        initial_epoch = 0

    #writer = SummaryWriter('D:/steep_training/ski-race/balanced/log/')
    
    model = model.to(device)
    
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
                
                ps = torch.exp(logps)
                max_preds = torch.max(ps, 1)[1]
                max_true = torch.max(local_labels, 1)[1]
                batch_acc = max_preds.eq(max_true).sum().item()

                print('Batch accuracy: ', batch_acc / 16)
                print('Batch loss: ', running_loss / 16)

                running_loss = 0.0

            if (curr_batch % 1250 == 0):

                checkpoint_state = {'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer: ', optimizer.state_dict()
                        }

                save_checkpoint(checkpoing_state)
                            
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

        checkpoint_state = {'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer: ', optimizer.state_dict()
                        }

        save_checkpoint(checkpoing_state)
 
        
if __name__ == '__main__':
    main(True)
