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

from reptile import Reptile, FOML
from util import save_checkpoint, plot_accuracy

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
          cuda=False,
          pin_memory=False,
          foml=False):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    reptile_fn=FOML if foml else Reptile
    reptile = reptile_fn(num_classes, model_state, op_state, cuda, pin_memory)

    train_writer = SummaryWriter(os.path.join(save_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(save_dir, 'test'))
    accuracy_tracking =  pd.DataFrame(columns=['meta_iteration','train_accuracy','test_accuracy']) 
    
    for i in tqdm.trange(1, meta_iters+1):
        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        
        reptile.train_step(dataset=train_set, num_classes=num_classes, num_shots=train_shots,
                             inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                             meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)
        
        if i % eval_interval == 0:
            accuracies = []
            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                accuracy = reptile.evaluate(dataset=dataset, num_classes=num_classes, num_shots=num_shots,
                                               inner_batch_size=eval_inner_batch_size,
                                               inner_iters=eval_inner_iters, transductive=transductive)
                accuracies.append(accuracy)
                writer.add_scalar('accuracy', accuracy, i)      
            accuracy_tracking.loc[len(accuracy_tracking)] = [i, accuracies[0], accuracies[1]]
            print('\nbatch {} - Accuracy moving averages: train={:2.2%}, test={:2.2%}, step_size={:.5f}'.format(i, accuracy_tracking.iloc[-100:]['train_accuracy'].mean(), accuracy_tracking.iloc[-100:]['test_accuracy'].mean(), cur_meta_step_size))
        if i % 2500 == 0 or i == meta_iters:
            save_checkpoint(reptile.model_state, reptile.op_state, save_dir, i, cur_meta_step_size, accuracy_tracking)
        if time_deadline is not None and time.time() > time_deadline:
            break
        
    train_writer.close()
    test_writer.close()
    accuracy_tracking.to_pickle(os.path.join(save_dir, r'accuracies.pkl'))
    plot_accuracy(accuracy_tracking, window=1000, filepath=os.path.join(save_dir, r'accuracy.pdf'), title='MiniimageNet 5-shot 5-way')
