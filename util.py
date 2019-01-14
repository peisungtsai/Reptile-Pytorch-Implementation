#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 09:36:06 2018

@author: peisungtsai
"""
import os
import re
import torch
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def save_checkpoint(model_state, op_state, check_dir, meta_iteration):
    state = {'meta_iteration': meta_iteration, 
             'state_dict': model_state,
             'optimizer': op_state}
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
        meta_iteration = checkpoint['meta_iteration']
        model_state = checkpoint['state_dict']
        op_state = checkpoint['optimizer']
        print("\n=> loaded checkpoint '{}' (meta_iteration {})\n"
                  .format(latest_file, checkpoint['meta_iteration']))
    else:
        print("\n=> no checkpoint found at '{}'".format(check_dir))

    return model_state, op_state, meta_iteration


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

def plot_accuracy(data, x, y, window, filepath, title=''):
    mean = pd.DataFrame(data[y].rolling(window).mean()).rename(index=str, columns={y: 'average'})
    std = pd.DataFrame(data[y].rolling(window).std()).rename(index=str, columns={y: 'std'})
    mean = mean.reset_index()
    std = std.reset_index()
    data = data.reset_index()
    df_plot = data[[x, y]].join(mean[['average']]).join(std[['std']])
    pdf = PdfPages(filepath) 
    
    chart = df_plot.plot(x=x, y=['average'], title=title)
    fig = chart.get_figure()
    pdf.savefig(fig)
    
    chart = df_plot.plot(x=x, y=['std'], title=title)
    fig = chart.get_figure()
    pdf.savefig(fig)  
    pdf.close()
    
#save_dir = '/Users/peisungtsai/supervised-reptile-pytorch/ckpt_m55'
#accuracies = pd.read_pickle(os.path.join(save_dir, r'accuracies.pkl'))
#plot_accuracy(data=accuracies, x='meta_iteration', y='test_accuracy', window=100, filepath=os.path.join(save_dir, r'accuracy.pdf'), title='MiniimageNet 5-shot 5-way')

