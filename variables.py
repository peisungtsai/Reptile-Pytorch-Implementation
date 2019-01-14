"""
Tools for manipulating sets of variables.
"""

import numpy as np
import torch

#import tensorflow as tf
#
#def subtract_vars(var_seq_1, var_seq_2):
#    """
#    Subtract one variable sequence from another.
#    """
#    return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]
#
#def add_vars(var_seq_1, var_seq_2):
#    """
#    Add two variable sequences.
#    """
#    return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]
#
#def scale_vars(var_seq, scale):
#    """
#    Scale a variable sequence.
#    """
#    return [v * scale for v in var_seq]
#
#def weight_decay(rate, variables=None):
#    """
#    Create an Op that performs weight decay.
#    """
#    if variables is None:
#        variables = tf.trainable_variables()
#    ops = [tf.assign(var, var * rate) for var in variables]
#    return tf.group(*ops)


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

#def import(model, state_dict, cuda):
#    model.load_state_dict(state_dict)
#    if cuda:
#        model.cuda()
#    return model