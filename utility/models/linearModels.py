# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:50:41 2019

@author: kretz01
"""
import torch
import torch.nn as nn 

class hidden1(nn.Module):
    def __init__(self, nb_features, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super(hidden1, self).__init__()
        
        self.linear1 = nn.Linear(1, nb_features)
        
        self.linear2_mu = nn.Linear(nb_features, 1)
        self.linear2_sigma = nn.Linear(nb_features, 1)
        
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus(beta=1, threshold=20)
    
    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        mu = self.linear2_mu(x1)    
        sigma = self.softplus(self.linear2_sigma(x1)) + 1e-6
        return mu, sigma