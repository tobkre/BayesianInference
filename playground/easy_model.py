# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:24:29 2019

@author: kretz01
"""
import torch
import torch.nn as nn

class hidden1(nn.Module):
    def __init__(self, nb_features):
        super(hidden1, self).__init__()
        
        self.linear1 = nn.Linear(1, nb_features)
        
        self.linear2_mu = nn.Linear(nb_features, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        mu = self.linear2_mu(x1)    
        return mu