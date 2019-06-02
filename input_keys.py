# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:03:49 2019

@author: Max
"""

from directkeys import PressKey, ReleaseKey



def find_key(one_hot, trimmed_one_hot_dict):
    
    for key in trimmed_one_hot_dict:
        if (trimmed_one_hot_dict[key] == one_hot):
            return key
        
    print('Error. Key not found.')
    
    
def manage_keys(key, prev_key, dx_key_dict):
    
    if (key == prev_key):
        return
    
    keep_W = (('W' in key) and ('W' in prev_key))

    if (prev_key != 'nk'):
        
        if ('space') in prev_key:
            ReleaseKey(dx_key_dict['space'])
        if ('shift') in prev_key:
            ReleaseKey(dx_key_dict['shfit'])

        for char in prev_key:
            
            if char.isupper():
                
                if ((char == 'W') and keep_W):
                    continue
                else:
                    ReleaseKey(dx_key_dict[char])
                
    if (key != 'nk'):
        
        if ('space') in key:
            PressKey(dx_key_dict['space'])
        if ('shift') in key:
            PressKey(dx_key_dict['shfit'])
            
        for char in key:
            
            if (char.isupper()):
                if ((char == 'W') and keep_W):
                    continue
                else:
                    PressKey(dx_key_dict[char])
                    
                    
def act_on_prediction(one_hot, trimmed_one_hot_dict, prev_key, dx_key_dict):
    
    key = find_key(one_hot, trimmed_one_hot_dict)
    manage_keys(key, prev_key, dx_key_dict)
    
    return key