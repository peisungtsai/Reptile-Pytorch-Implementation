"""
Models for supervised meta-learning.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from args import argument_parser
from adamW import AdamW

args = argument_parser().parse_args() 
    
class MiniimagenetModel(nn.Module):
    """
    A model for Mini Imagenet classification.
    """    
    
    def __init__(self, num_classes):
        super(MiniimagenetModel, self).__init__()
        
        self.conv = nn.Sequential(
            # 84 x 84 - 3
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.ReLU(True),

            # 42 x 42 - 32
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.ReLU(True),

            # 21 x 21 - 32
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.ReLU(True),

            # 11 x 11 - 32
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        # 6 x 6 x 32 = 1152
        self.classifier = nn.Linear(1152, num_classes)

    def forward(self, x):
        out = x.view(-1, 3, 84, 84)
        out = self.conv(out)
        out = out.view(len(out), -1)
        out = self.classifier(out)
        return F.log_softmax(out, 1)    

#class MiniimagenetModel(nn.Module):
#    """
#    A model for Mini Imagenet classification.
#    """    
#    
#    def __init__(self, num_classes):
#        super(MiniimagenetModel, self).__init__()
#        
#        self.conv = nn.Sequential(
#            # 96 x 96 - 3
#            nn.Conv2d(3, 32, 3, 1, 1),
#            nn.BatchNorm2d(32),
#            nn.MaxPool2d(2, stride=2, padding=0),
#            nn.ReLU(True),
#
#            # 48 x 48 - 32
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.BatchNorm2d(32),
#            nn.MaxPool2d(2, stride=2, padding=0),
#            nn.ReLU(True),
#
#            # 24 x 24 - 32
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.BatchNorm2d(32),
#            nn.MaxPool2d(2, stride=2, padding=0),
#            nn.ReLU(True),
#
#            # 12 x 12 - 32
#            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.BatchNorm2d(32),
#            nn.MaxPool2d(2, stride=2, padding=0),
#            nn.ReLU(True)
#        )
#        
#        # 6 x 6 x 32 = 1152
#        self.classifier = nn.Linear(1152, num_classes)
#
#    def forward(self, x):
#        out = x.view(-1, 3, 96, 96)
#        out = self.conv(out)
#        out = out.view(len(out), -1)
#        out = self.classifier(out)
#        return F.log_softmax(out, 1)
    
    
class OmniglotModel(nn.Module):
    """
    A model for Omniglot classification.
    """    
    def __init__(self, num_classes):
        super(OmniglotModel, self).__init__()

        self.conv = nn.Sequential(
            # 28 x 28 - 1
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 14 x 14 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 7 x 7 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 4 x 4 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 2 x 2 - 64
        )
        # 2 x 2 x 64 = 256
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        out = x.view(-1, 1, 28, 28)
        out = self.conv(out)
        out = out.view(len(out), -1)
        out = self.classifier(out)
        return F.log_softmax(out, 1)


def get_optimizer(model, state=None):
    if args.sgd:
#        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0, 0.999), weight_decay=args.weight_decay, amsgrad=False)   
#        optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0, 0.999), weight_decay=args.weight_decay, amsgrad=False) 
        
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer
    

def clone_model(num_classes, model_state=None):   
    model_clone = MiniimagenetModel(num_classes)
    
    if args.cuda:
        if args.parallel:
            model_clone = nn.DataParallel(model_clone, device_ids=[0, 1]).cuda()
        else:
            model_clone.cuda()    
    
    if model_state is not None:
        model_clone.load_state_dict(model_state)
            
    return model_clone
    
    
def predict_label(prob):
    __, argmax = prob.max(1)
    return argmax
    
def truncated_normal_(tensor, mean=0.0, std=0.1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    
def get_loss(prediction, labels):
    loss = nn.NLLLoss()
    return loss(prediction, labels)


#def clone(num_classes, model_state, op_state, cuda):   
#    model_clone = OmniglotModel(num_classes)
#    model_dict = model_clone.state_dict()    
#    
#    #Get trainable new vars.
#    trainable=[]    
#    for name, param in model_clone.named_parameters():
#        if param.requires_grad and "classifier" not in name:
#            trainable.append(name)  
#            
#    model_state = {name: model_state[name] for name in trainable}       
#    
#    model_dict.update(model_state) 
#    model_clone.load_state_dict(model_dict)
#    model_clone.cuda() if cuda else None
#    
#    return model_clone, get_optimizer(model_clone, state=op_state)
