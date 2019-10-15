# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:09:55 2019

@author: kretz01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class MyNet(nn.Module):
    def __init__(self, n_in, n_out):
        super(MyNet, self).__init__()
        
        self.temp = 1./10.
        
        self.inlayer = nn.Linear(n_in, 1024)
        self.inlayer.p_par = nn.Parameter(torch.tensor([0.0]))
        self.hidden1 = nn.Linear(1024, 1024)
        self.hidden1.p_par = nn.Parameter(torch.tensor([0.0]))
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden2.p_par = nn.Parameter(torch.tensor([0.0]))
#        self.hidden3 = nn.Linear(1024, 1024)
#        self.hidden3.p_par = nn.Parameter(torch.tensor([0.0]))
        self.outlayer = nn.Linear(1024, n_out)
        self.outlayer.p_par = nn.Parameter(torch.tensor([0.0]))
        
        self.linear_output=True
    
    @staticmethod
    def p(p_par, scale=0.5):
#        set_trace()
        return torch.sigmoid(p_par* scale)
    
    def get_size(self, x):
        return nn.prod(x.shape[1:])
    
    def concrete_drop(self, shape, p):        
#        u = Uniform(0,1)
        repeat_shape = [shape[0]] + len(shape[1:]) * [1]
        u = torch.rand(shape[1:]).repeat(repeat_shape)
        drop_prob = torch.log(p)-torch.log(1-p)+torch.log(u)-torch.log(1-u)
        return torch.sigmoid(drop_prob/self.temp)    
    
    def concretify(self, layer):
        def concretified_layer(x):
            dropped_x = x * (1-self.concrete_drop(x.shape, self.p(layer.p_par)))
#            set_trace()
            out = layer(dropped_x)
            return out
        return concretified_layer
    
    def forward(self, x):
#        batch_size = x.shape[0]
        x = torch.relu(self.concretify(self.inlayer)(x))
        x = torch.relu(self.concretify(self.hidden1)(x))
        x = torch.relu(self.concretify(self.hidden2)(x))
#        x = torch.relu(self.concretify(self.hidden3)(x))
        x = self.concretify(self.outlayer)(x)
        if self.linear_output:
            out = x
        else:
            out = F.log_softmax(x, dim=1)
        return out