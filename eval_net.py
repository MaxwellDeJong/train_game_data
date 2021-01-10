# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:10:06 2019

@author: Max
"""
import pickle
import cv2
import time
import torch
import numpy as np
from torchvision import transforms
import torch.backends.cudnn as cudnn

from network_utils import load_normalization_stats
from load_resnet import load_resnet50
from grabscreen import grab_screen
from getkeys import key_check
from directkeys import get_key_dict, ReleaseAllKeys
from input_keys import act_on_prediction
from train_net import load_checkpoint
import time


def load_one_hot_dict(dirname='ski-race'):
        
    with open('E:/steep_training/' + dirname + '/balanced/one_hot_dict.pkl', 'rb') as handle:
        one_hot_dict = pickle.load(handle)
        
    return one_hot_dict


def read_raw_screen():
    
    title_bar_offset = 30
    
    screen = grab_screen(region=(0, title_bar_offset, 1260, 710))
    screen = cv2.resize(screen, (320, 180))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    
    return screen


def read_screen(means, stds):
    
    screen = read_raw_screen()
        
    frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
            ])
    
    torch_screen = frame_transform(screen)
    
    return torch_screen

def form_normalization_tensor(dirname='ski-race'):
    
    with open('E:/steep_training/' + dirname + '/balanced/significant_count_dict.pkl', 'rb') as p:
        counts = pickle.load(p)
        
    with open('E:/steep_training/' + dirname + '/balanced/one_hot_dict.pkl', 'rb') as p:
        one_hot = pickle.load(p)
    
    normalization_arr = torch.zeros(len(counts.keys()), dtype=torch.float)    
    for key in counts.keys():
        normalization_arr += counts[key] * torch.tensor(one_hot[key], dtype=torch.float)
        
    normalization_arr /= torch.sum(normalization_arr)
    
    #return torch.tensor([1., 1., 1., 1., 1., 1.], dtype=torch.float)
    return torch.tensor([1., 1., 1.], dtype=torch.float)
    #return normalization_arr      
    

def prediction_to_one_hot(pred, normalization_counts):
    
    one_hot = torch.zeros(pred.numel(), dtype=torch.int32)
    max_idx = (pred * normalization_counts).argmax().item()
    
    #max_idx = pred.argmax().item()
    one_hot[max_idx] = 1
    
    return one_hot
  
    
def load_model(model, dirname='ski-race'):

    filename = 'E:/steep_training/' + dirname + '/balanced/model/resnet50.pth'

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])

    
def eval_net(dirname='ski-race', dropout=False, weight_decay=False, large_decay=False, epoch=9):
        
    (means, stds) = load_normalization_stats(dirname=dirname)

    dx_key_dict = get_key_dict()
    one_hot_dict = load_one_hot_dict(dirname=dirname)
    
    device = torch.device('cuda:0')
    cudnn.benchmark = True
    cudnn.enabled = True
    
    model = load_resnet50(use_custom=True, dropout=dropout, dirname=dirname)       
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    load_model(model, dirname=dirname)
    
    #load_checkpoint(model, optimizer, dirname=dirname, \
    #                                weight_decay=weight_decay, dropout=dropout,
    #                                epoch=epoch, large_decay=large_decay)
    
    model.eval()
    torch.set_grad_enabled(False)
    
    normalization_counts = form_normalization_tensor(dirname=dirname)
    normalization_counts = normalization_counts.to(device)
        
#    time.sleep(5)
#    print('Done sleeping!')attawwawawtw
    
    paused = False
    keys = key_check()
    
    prev_key = ''
    
    start_time = time.time()
    frames_analyzed = 0
    
    while True:   
        if not paused:
            
#            initial_time = time.time()
#            torch.cuda.synchronize()
            screen = read_screen(means, stds)
            
            screen = screen.to(device)
            
#            torch.cuda.synchronize()
#            screen_time = time.time()
#            torch.cuda.synchronize()
            
            screen = screen[None]
            
            pred = model(screen)
            
#            torch.cuda.synchronize()
#            pred_time = time.time()
#            torch.cuda.synchronize()
            
            one_hot = prediction_to_one_hot(pred, normalization_counts)
            one_hot.cpu()
            prev_key = act_on_prediction(one_hot.tolist(), one_hot_dict, prev_key, dx_key_dict)
            
#            final_time = time.time()
            
#            print('Processing of single frame took ', final_time - initial_time)
#            print('\t Screen read, process, and transfer time: ', screen_time - initial_time)
#            print('\t Inference time: ', pred_time - screen_time)
#            print('\t Time to act: ', final_time - pred_time)
            
        else:
            if 'X' in keys:
                break
        
        if 'T' in keys:
            
            ReleaseAllKeys(dx_key_dict)

            if paused:

                paused = False
                print('unpaused!')
                time.sleep(0.1)

            else:
                print('Pausing!')
                paused = True
                time.sleep(0.1)
                
        if not paused:
        
            frames_analyzed += 1
            
            if (frames_analyzed % 10 == 9):
                print('Last ten frames averaged ', 10 / (time.time() - start_time), ' fps')
                start_time = time.time()
        
        keys = key_check()
        
if __name__ == '__main__':
    eval_net(dirname='ski-race3', dropout=False, weight_decay=True, large_decay=False, epoch=9)