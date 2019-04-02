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

from variables import Variable_, interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars
from models import get_loss, predict_label, clone_model, get_optimizer

class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, num_classes, model_state=None, op_state=None, cuda=False, pin_memory=False):      
        if model_state == None:
            model = clone_model(num_classes)
            self.model_state = model.state_dict()   
        else:
            self.model_state = model_state   

        self.op_state = op_state
        self.cuda = cuda
        self.pin_memory = pin_memory
        
    def train_step(self,
                   dataset,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   meta_step_size,
                   meta_batch_size):
        """
        Perform a Reptile training step.
    
        Args:
          dataset: object contains images to be trained.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        new_vars = []
        for _ in range(meta_batch_size):
            model = clone_model(num_classes, self.model_state)
            optimizer = get_optimizer(model, self.op_state)            
            model.train()    
            mini_train, _ = dataset.get_random_task_split(num_classes, train_shots=num_shots, test_shots=0)
            """
            Sampling without replacement
            """
            sampler=RandomSampler(mini_train, replacement=False)
            train_loader = DataLoader(mini_train, batch_size=inner_batch_size, sampler=sampler, drop_last=True, pin_memory=self.pin_memory)          
            for _, (inputs, labels) in enumerate(train_loader):            
                inputs, labels = Variable_(inputs, labels, self.cuda)
                prediction = model(inputs)
                loss = get_loss(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            new_vars.append(model.state_dict())
            self.op_state = optimizer.state_dict()
        new_vars = average_vars(new_vars)
        self.model_state = interpolate_vars(self.model_state, new_vars, meta_step_size)
        
        
    def evaluate(self,
                 dataset,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 transductive):
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
    
        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        model_clone = clone_model(num_classes, self.model_state)
        optimizer_clone = get_optimizer(model_clone, self.op_state)
        model_clone.train()    
        train, test = dataset.get_random_task_split(num_classes, train_shots=num_shots, test_shots=1)
        """
        Sampling with replacement
        """
        sampler=RandomSampler(train, replacement=True, num_samples=inner_batch_size*inner_iters)
        train_loader = DataLoader(train, batch_size=inner_batch_size, sampler=sampler, pin_memory=self.pin_memory)
        for _, (inputs, labels) in enumerate(train_loader):       
            inputs, labels = Variable_(inputs, labels, self.cuda)
            prediction = model_clone(inputs)
            loss = get_loss(prediction, labels)
            optimizer_clone.zero_grad()
            loss.backward()
            optimizer_clone.step()      
        return self._test_predictions(model_clone, train, test, transductive, self.cuda)
    
    
    def _test_predictions(self, model, train, test, transductive, cuda):
        model.eval()
        if transductive:
            test_iter = iter(DataLoader(test, batch_size=len(test), shuffle=True, pin_memory=self.pin_memory))
            inputs, labels = next(test_iter)  
            inputs, labels = Variable_(inputs, labels, self.cuda)    
            prediction = model(inputs)
            argmax = predict_label(prediction)
            accuracy = (argmax == labels).float().mean()  
        else:
            train_iter = iter(DataLoader(train, batch_size=len(train), shuffle=True, pin_memory=self.pin_memory))
            train_inputs, train_labels = next(train_iter) 
            train_inputs, train_labels = Variable_(train_inputs, train_labels, self.cuda)
            test_iter = iter(DataLoader(test, batch_size=len(test), shuffle=True, pin_memory=self.pin_memory))
            test_inputs, test_labels = next(test_iter) 
            test_inputs, test_labels = Variable_(test_inputs, test_labels, self.cuda) 
            
            predict_list =[]
            for i in range(test_inputs.size()[0]):
                select = test_inputs.select(0, i).unsqueeze(0)  
                input_temp = torch.cat((select, train_inputs), 0)
                predict = model(input_temp)
                predict_list.append(predict[0])
                
            prediction=torch.stack(predict_list)  
            argmax = predict_label(prediction) 
            accuracy = (argmax == test_labels).float().mean() 
             
        return accuracy.data.item()


class FOML(Reptile):
    """
    A basic implementation of "first-order MAML" (FOML).

    FOML is similar to Reptile, except that you use the
    gradient from the last mini-batch as the update
    direction.

    There are two ways to sample batches for FOML.
    By default, FOML samples batches just like Reptile,
    meaning that the final mini-batch may overlap with
    the previous mini-batches.
    Alternatively, if tail_shots is specified, then a
    separate mini-batch is used for the final step.
    This final mini-batch is guaranteed not to overlap
    with the training mini-batches.
    """        
    def train_step(self,
                   dataset,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   meta_step_size,
                   meta_batch_size):
        """
        Perform a Reptile training step.
    
        Args:
          dataset: object contains images to be trained.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        
        updates = []
        for _ in range(meta_batch_size):
            model = clone_model(num_classes, self.model_state)
            optimizer = get_optimizer(model, self.op_state)  
            model.train()            
            mini_train, _ = dataset.get_random_task_split(num_classes, train_shots=num_shots, test_shots=0)
            """
            Sampling without replacement
            """
            train_loader = DataLoader(mini_train, batch_size=inner_batch_size, drop_last=True, shuffle=True, pin_memory=self.pin_memory)           
            for _, (inputs, labels) in enumerate(train_loader):  
                inputs, labels = Variable_(inputs, labels, self.cuda)
                last_backup = model.state_dict()
                prediction = model(inputs)
                loss = get_loss(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            updates.append(subtract_vars(model.state_dict(), last_backup))
            self.op_state = optimizer.state_dict()
        update = average_vars(updates)
        self.model_state = add_vars(self.model_state, scale_vars(update, meta_step_size))
        self.op_state = optimizer.state_dict()
        
        
