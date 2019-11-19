# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:51:01 2019

@author: kretz01
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:25:06 2019

@author: kretz01
"""
import torch
import torch.nn as nn 
import numpy as np

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-6, 
                 init_min=0.5, init_max=0.5):
        super(ConcreteDropout, self).__init__()
        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
#        init_min = np.log(init_min) - np.log(1-init_min)
#        init_max = np.log(init_max) - np.log(1-init_max)
        
#        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
#        self.p_logit.requires_grad = True
        
    def forward(self, x, layer, p_logit):
        p = torch.sigmoid(p_logit)
        
        out = layer(self._concrete_dropout(x, p))
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square/(1-p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1.-p) * torch.log(1.-p)
        
        input_dim = x[0].numel()
        dropout_regularizer = self.dropout_regularizer * input_dim
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
    
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1
        
        unif_noise = torch.rand_like(x)
        drop_prob = (torch.log(p+eps) - torch.log(1-p+eps) + torch.log(unif_noise+eps) - torch.log(1-unif_noise+eps))
        drop_prob = torch.sigmoid(drop_prob/temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1- p
        
        x = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x

class myNet(nn.Module):
    def __init__(self, nb_features, weight_regularizer=1e-6, dropout_regularizer=1e-5, 
                 p_init=2.4e-02, train_p = False):
        super(myNet, self).__init__()
        
        self.linear1 = nn.Linear(1, nb_features)
        
        self.mu = nn.Linear(nb_features, 1)
        self.sig = nn.Linear(nb_features, 1)
#        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer, 
#                                          dropout_regularizer=dropout_regularizer)
        self.conc_dropmu = ConcreteDropout(weight_regularizer=weight_regularizer, 
                                          dropout_regularizer=dropout_regularizer)
        self.conc_dropsig = ConcreteDropout(weight_regularizer=weight_regularizer, 
                                          dropout_regularizer=dropout_regularizer)
        self.relu = nn.ReLU()
        p_init = np.log(p_init) - np.log(1-p_init)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(p_init, p_init))
        self.p_logit.requires_grad = train_p
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
    def forward(self, x):
        regularization = torch.empty(3, device=x.device)
        x1 = self.relu(self.linear1(x))
        
        p = torch.sigmoid(self.p_logit)
        sum_of_square = 0
        for param in self.linear1.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square/(1-p)
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1.-p) * torch.log(1.-p)
        
        input_dim = x[0].numel()
        dropout_regularizer = self.dropout_regularizer * input_dim
        
        regularization[0] = weights_regularizer + dropout_regularizer
        
        mu, regularization[1] = self.conc_dropmu(x1, self.mu, self.p_logit)
        
        sig, regularization[2] = self.conc_dropsig(x1, self.sig, self.p_logit)
        return mu, sig, regularization.sum()