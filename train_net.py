# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:51:50 2019

@author: Max
"""
from load_resnet import load_resnet50
from network_utils import load_generators
import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import time
import os


def save_checkpoint(state, weight_decay=False, dropout=False, dirname='ski-race', large_decay=False):
        
    filename = get_modelname(weight_decay, dropout, state['epoch'], dirname=dirname, large_decay=large_decay)
    model_dir = get_network_dir(weight_decay, dropout, dirname=dirname, model=True, large_decay=large_decay)
    
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    torch.save(state, filename)


def get_network_dir(weight_decay, dropout, dirname='ski-race', model=True, large_decay=False):

    if model:
        prefix = 'model/'
    else:
        prefix = 'log/'
        
    network_dir = 'D:/steep_training/' + dirname + '/balanced/' + prefix
    
    if weight_decay and dropout:
        if large_decay:
            network_dir = network_dir + 'dropout__weight_decay_lg/'
        else:
            network_dir = network_dir + 'dropout__weight_decay/'
    elif dropout:
        network_dir = network_dir + 'dropout/'
    elif weight_decay:
        if large_decay:
            network_dir = network_dir + 'weight_decay_lg/'
        else:
            network_dir = network_dir + 'weight_decay/'
    else:
        network_dir = network_dir + 'vanilla/'
        
    return network_dir
    
    
def get_modelname(weight_decay, dropout, epoch, dirname='ski-race', large_decay=False):
    
    model_dir = get_network_dir(weight_decay, dropout, dirname=dirname, model=True, large_decay=large_decay)
    return model_dir + 'resnet50_epoch%d.pth' % epoch


def load_checkpoint(model, optimizer, dirname='ski-race', weight_decay=False, dropout=False, epoch=-1, large_decay=False):

    if epoch >= 0:
        filename = get_modelname(weight_decay, dropout, epoch, dirname=dirname, large_decay=large_decay)

    else:
        next_filename = get_modelname(weight_decay, dropout, 0, dirname=dirname, large_decay=large_decay)
        idx = 0
        while os.path.exists(next_filename):
            filename = next_filename
            next_filename = get_modelname(weight_decay, dropout, idx+1, dirname=dirname, large_decay=large_decay)
            idx += 1

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']+1


def main(load_model, weight_decay=False, dropout=False, dirname='ski-race', epoch=-1, large_decay=False):

    device = torch.device('cuda:0')
    cudnn.benchmark = True
    
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 0}
    
    (training_generator, validation_generator) = load_generators(params, dirname=dirname)

    criterion = nn.CrossEntropyLoss()
    
    model = load_resnet50(use_custom=True, dropout=dropout, dirname=dirname)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    if weight_decay:
        if large_decay:
            optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)
        else:
            optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters())
        
    model = model.to(device)

    if (load_model):
        initial_epoch = load_checkpoint(model, optimizer, dirname=dirname, \
                                    weight_decay=weight_decay, dropout=dropout,
                                    epoch=epoch, large_decay=large_decay)

    else:
        initial_epoch = 0        

    writer_dir = get_network_dir(weight_decay, dropout, dirname=dirname, model=False, large_decay=large_decay)
    writer = SummaryWriter(writer_dir)
 
    max_epoch = 10
    
    time0 = time.time()
    
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
            
            if (curr_batch % 1500 == 0):
                print('Have analyzed ', curr_batch * 64, ' frames in epoch ', epoch)
                print('Total runtime: ', (time.time() - time0) / 60)
                
                ps = torch.exp(logps)
                max_preds = torch.max(ps, 1)[1]
                max_true = torch.max(local_labels, 1)[1]
                batch_acc = max_preds.eq(max_true).sum().item()

                print('Batch accuracy: ', batch_acc / 64)
                print('Batch loss: ', running_loss / 64)

                writer.add_scalar('Train/Loss', running_loss / 64., curr_batch * 1500)
                writer.add_scalar('Train/Accuracy', batch_acc / 64., curr_batch * 1500)

                running_loss = 0.0

#            if (curr_batch % 2000 == 0):
#
#                checkpoint_state = {'epoch': epoch,
#                        'state_dict': model.state_dict(),
#                        'optimizer': optimizer.state_dict()
#                        }
#
#                save_checkpoint(checkpoint_state)
#                            
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

            writer.add_scalar('Val/Loss', cum_loss / len(validation_generator), epoch)
            writer.add_scalar('Val/Accuracy', accuracy / len(validation_generator), epoch)

        checkpoint_state = {'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }

        save_checkpoint(checkpoint_state, weight_decay=weight_decay, \
                        dropout=dropout, dirname=dirname, large_decay=large_decay)

    writer.close()
 
        
if __name__ == '__main__':
    main(True, weight_decay=True, dropout=True, dirname='wing-suit', large_decay=False)
    main(False, weight_decay=True, dropout=True, dirname='wing-suit', large_decay=True)
