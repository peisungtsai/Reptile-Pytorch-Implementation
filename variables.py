"""
Tools for manipulating sets of variables.
"""

import numpy as np
import torch

def Variable_(inputs, labels, cuda=False):
    '''
    Make variable cuda depending on the arguments
    '''
    if cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
    return inputs, labels

def average_vars(dict_list):
    """
    Averrage of list of state_dicts.
    """
    for param_tensor in dict_list[0]:
        for i in range(1, len(dict_list)):
            dict_list[0][param_tensor] = dict_list[0][param_tensor] + dict_list[i][param_tensor]
        dict_list[0][param_tensor] = dict_list[0][param_tensor]/len(dict_list)
    
    average_var = dict_list[0]

    return average_var

def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    for param_tensor in new_vars:
        new_vars[param_tensor]  = old_vars[param_tensor] + (new_vars[param_tensor] - old_vars[param_tensor]) * epsilon
    return new_vars

def subtract_vars(new_vars, old_vars):
    """
    Subtract one variable sequence from another.
    
    """
    for param_tensor in new_vars:
        new_vars[param_tensor]  = new_vars[param_tensor] - old_vars[param_tensor] 
    return new_vars

def add_vars(var_1, var_2):
    """
    Add two variable sequences.
    """
    for param_tensor in var_1:
        var_1[param_tensor]  = var_1[param_tensor] + var_2[param_tensor] 
    return var_1

def scale_vars(var, epsilon):
    """
    Scale a variable sequence.
    """
    for param_tensor in var:
        var[param_tensor]  = var[param_tensor] * epsilon
    return var


#def import(model, state_dict, cuda):
#    model.load_state_dict(state_dict)
#    if cuda:
#        model.cuda()
#    return model