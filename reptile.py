"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import random
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from supervised_reptile.variables import (interpolate_vars, average_vars, Variable_)
from supervised_reptile.models import get_loss, predict_label, clone_model, get_optimizer

class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, num_classes, model_state=None, op_state=None):
        if model_state == None:
            model, optimizer = clone_model(num_classes)
            self.model_state = model.state_dict()   
            self.op_state = optimizer.state_dict() 
        else:
            self.model_state = model_state   
            self.op_state = op_state
            
    def train_step(self,
                   dataset,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   cuda,
                   meta_step_size,
                   meta_batch_size):
        """
        Perform a Reptile training step.
    
        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        
        new_vars = []
        for _ in range(meta_batch_size):
            model, optimizer = clone_model(num_classes, self.model_state, self.op_state)
            model.train()            
            mini_train, _ = dataset.get_random_task_split(num_classes, train_shots=num_shots, test_shots=0)
            train_loader = DataLoader(mini_train, batch_size=inner_batch_size, shuffle=True, pin_memory=cuda)           
            for _, (inputs, labels) in enumerate(train_loader):  
                inputs, labels = Variable_(inputs, labels, cuda)
                prediction = model(inputs)
                loss = get_loss(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            new_vars.append(model.state_dict())
        new_vars = average_vars(new_vars)
        self.model_state = interpolate_vars(self.model_state, new_vars, meta_step_size)
        self.op_state = optimizer.state_dict()

    
    def evaluate(self,
                 dataset,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement,
                 cuda):
        """
        Run a single evaluation of the model.
    
        Samples a few-shot learning task and measures
        performance.
    
        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
    
        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        model, optimizer = clone_model(num_classes, self.model_state, self.op_state)
        model.train()    
        train_set, test_set = dataset.get_random_task_split(num_classes, train_shots=num_shots, test_shots=1)
        sampler=RandomSampler(train_set, replacement=True, num_samples=inner_batch_size*inner_iters)
        train_loader = DataLoader(train_set, batch_size=inner_batch_size, sampler=sampler, pin_memory=cuda)
        for _, (inputs, labels) in enumerate(train_loader):       
            inputs, labels = Variable_(inputs, labels, cuda)
            prediction = model(inputs)
            loss = get_loss(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()      
        return self._test_predictions(model, train_set, test_set, cuda)
    
    
    def _test_predictions(self, model, train_set, test_set, cuda):
        model.eval()
        test_iter = iter(DataLoader(test_set, batch_size=len(test_set), shuffle=True)) 
        inputs, labels = next(test_iter)  
        inputs, labels = Variable_(inputs, labels, cuda)    
        prediction = model(inputs)
        argmax = predict_label(prediction)
        accuracy = (argmax == labels).float().mean()  
        return accuracy.data.item()
