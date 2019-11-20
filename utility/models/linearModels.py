# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:50:41 2019

@author: kretz01
"""
import torch
import torch.nn as nn 

class hidden1(nn.Module):
    def __init__(self, n_hidden, weight_regularizer=1e-4, dropout_regularizer=1e-5, noise='homoscedastic'):
        super(hidden1, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.noise = noise
        
        self.linear1 = nn.Linear(1, n_hidden) 
        self.mu = nn.Linear(n_hidden, 1)
        if noise=='homoscedastic':
            self.logvar = nn.Parameter(torch.FloatTensor([1e-1]))
        elif noise=='heteroscedastic':
            self.logvar = nn.Linear(n_hidden, 1)
        else:
            self.logvar = torch.FloatTensor([0.])
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        reg = self._regularize(self.linear1)
        mu = self.mu(x1)
        reg += self._regularize(self.mu)
        if self.noise=='heteroscedastic':
            logvar = self.logvar(x1)
#            reg += self._regularize(self.logvar)
        else:
            logvar = self.logvar * torch.ones_like(x)
        return mu, logvar, self.weight_regularizer * reg
    
    def _regularize(self, layer):
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square