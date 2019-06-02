# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:10:06 2019

@author: Max
"""
import pickle
import cv2
import time
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from network_utils import load_normalization_stats
from grabscreen import grab_screen
from getkeys import key_check
from directkeys import get_key_dict, ReleaseAllKeys
from input_keys import act_on_prediction


def load_one_hot_dict():
        
    with open('D:/steep_training/ski-race/balanced/one_hot_dict.pkl', 'rb') as handle:
        one_hot_dict = pickle.load(handle)
        
    return one_hot_dict


def load_model():
    
    model = torch.load('D:/steep_training/ski-race/balanced/model/inceptionresnetv2.pth')
    
    return model


def read_raw_screen():
    
    title_bar_offset = 30
    
    screen = grab_screen(region=(0, title_bar_offset, 1260, 710))
    screen = cv2.resize(screen, (512, 289))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    
    return screen


def read_screen(means, stds):
    
    screen = read_raw_screen()
    
    frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
            ])
    
    return frame_transform(screen)
    
    
def eval_net():
        
    (means, stds) = load_normalization_stats()
    
    dx_key_dict = get_key_dict()
    one_hot_dict = load_one_hot_dict()
    
    device = torch.device('cuda:0')
    cudnn.benchmark = True
    
    model = load_model()
    model = model.to(device)
    
    paused = False
    keys = key_check()
    
    prev_key = None
    
    while True:
        
        if not paused:
        
            screen = read_screen()
            _, one_hot = torch.max(model(screen), 1)
            prev_key = act_on_prediction(one_hot, one_hot_dict, prev_key, dx_key_dict)
            
        else:
            if 'X' in keys:
                break
        
        
        if 'T' in keys:
            
            ReleaseAllKeys(dx_key_dict)

            if paused:

                paused = False
                print('unpaused!')
                time.sleep(1)

            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
                
            keys = key_check()
        
if __name__ == '__main__':
    eval_net()