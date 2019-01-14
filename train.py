"""
Training helpers for supervised meta-learning.
"""

import os
import time
import tqdm
import re
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from supervised_reptile.reptile import Reptile
#from .variables import weight_decay
from supervised_reptile.util import save_checkpoint

# pylint: disable=R0913,R0914
def train(train_set,
          test_set,
          save_dir,
          model_state=None,
          op_state=None,
          num_classes=5,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=10,
          time_deadline=None,
          train_shots=None,
          transductive=False,
          cuda=False):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    reptile = Reptile(num_classes, model_state, op_state)

    train_writer = SummaryWriter(os.path.join(save_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(save_dir, 'test'))
    df_tracking =  pd.DataFrame(columns=['meta_iteration','train_accuracy','test_accuracy']) 
    
    for i in tqdm.trange(1, meta_iters+1):
        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        
        reptile.train_step(dataset=train_set, num_classes=num_classes, num_shots=(train_shots or num_shots),
                             inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                             replacement=replacement, cuda=cuda,
                             meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)
        
        if i % eval_interval == 0:
            accuracies = []
            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                accuracy = reptile.evaluate(dataset=dataset, num_classes=num_classes, num_shots=num_shots,
                                               inner_batch_size=eval_inner_batch_size,
                                               inner_iters=eval_inner_iters, replacement=replacement, cuda=cuda)
                accuracies.append(accuracy)
                writer.add_scalar('accuracy', accuracy, i)              
            print('\nbatch {}: train={:2.0%} test={:2.0%} step_size={:.5f}'.format(i, accuracies[0], accuracies[1], cur_meta_step_size))
            df_tracking.loc[len(df_tracking)] = [i, accuracies[0], accuracies[1]]
        if i % 1000 == 0 or i == meta_iters:
            save_checkpoint(reptile.model_state, reptile.op_state, save_dir, meta_iteration=i)
        if time_deadline is not None and time.time() > time_deadline:
            break
        
    train_writer.close()
    test_writer.close()
    df_tracking.to_pickle(os.path.join(save_dir, r'accuracies.pkl'))
    plot_accuracy(df_tracking, x='meta_iteration', y='test_accuracy', window=100, filepath=os.path.join(save_dir, r'accuracy.pdf'), title='')
