#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 09:36:06 2018

@author: peisungtsai
"""
import os
import re
import numpy as np
import scipy as sp
from scipy import stats

import torch
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def save_checkpoint(model_state, op_state, check_dir, meta_iteration, cur_meta_step_size, accuracy_tracking):
    state = {'meta_iteration': meta_iteration, 
             'state_dict': model_state,
             'optimizer': op_state,
             'cur_meta_step_size': cur_meta_step_size,
             'accuracy_tracking': accuracy_tracking
             }
    check_file = os.path.join(check_dir, '{}-{}.pth'.format(check_dir, meta_iteration))
    torch.save(state, check_file)

    
def load_checkpoint(check_dir):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    checkpoints = []
    for fname in os.listdir(check_dir):
        if len(fname.split('-')) > 1:
            s = re.findall(r'\d+', fname.split('-')[1])
            checkpoints.append((int(s[0]), fname))  
            
    latest_file = os.path.join(check_dir, max(checkpoints)[1])
    
    start_itr = 0
    if os.path.isfile(latest_file):
        print("\n=> loading checkpoint '{}'".format(latest_file))
        checkpoint = torch.load(latest_file)
        model_state = checkpoint['state_dict']
        op_state = checkpoint['optimizer']
        meta_iteration = checkpoint['meta_iteration']
        cur_meta_step_size = checkpoint['cur_meta_step_size']
        accuracy_tracking = checkpoint['accuracy_tracking']
        print("\n=> loaded checkpoint '{}' (meta_iteration {})\n"
                  .format(latest_file, checkpoint['meta_iteration']))
    else:
        print("\n=> no checkpoint found at '{}'".format(check_dir))

    return model_state, op_state, meta_iteration, cur_meta_step_size, accuracy_tracking


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories

def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), sp.stats.sem(data)
    me = se * sp.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, me

def plot_accuracy(data, window, filepath, title=''):
    test_mean = pd.DataFrame(data['test_accuracy'].rolling(window).mean()).rename(index=str, columns={'test_accuracy': 'Rolling Test Accuracy'})
    test_std = pd.DataFrame(data['test_accuracy'].rolling(window).std()).rename(index=str, columns={'test_accuracy': 'Rolling Test STD'})
    train_mean = pd.DataFrame(data['train_accuracy'].rolling(window).mean()).rename(index=str, columns={'train_accuracy': 'Rolling Train Accuracy'})
    train_std = pd.DataFrame(data['train_accuracy'].rolling(window).std()).rename(index=str, columns={'train_accuracy': 'Rolling Train STD'})    
    df_rolling = test_mean.join(test_std).join(train_mean).join(train_std)
    df_rolling = df_rolling.reset_index()
    data = data.reset_index()
    df_plot = data[['meta_iteration']].join(df_rolling)
    
    pdf = PdfPages(filepath) 
    
    chart = df_plot.plot(x='meta_iteration', y=['Rolling Test Accuracy'], title=title)
    fig = chart.get_figure()
    pdf.savefig(fig)
    
    chart = df_plot.plot(x='meta_iteration', y=['Rolling Train Accuracy','Rolling Test Accuracy'], title=title)
    fig = chart.get_figure()
    pdf.savefig(fig)
    
    chart = df_plot.plot(x='meta_iteration', y=['Rolling Train STD','Rolling Test STD'], title=title)
    fig = chart.get_figure()
    pdf.savefig(fig)  
    pdf.close()
    
#save_dir = '/Users/peisungtsai/supervised-reptile-pytorch/ckpt_m55'
#accuracies = pd.read_pickle(os.path.join(save_dir, r'accuracies.pkl'))
#plot_accuracy(data=accuracies, window=1000, filepath=os.path.join(save_dir, r'accuracy.pdf'), title='MiniimageNet 5-shot 5-way')

