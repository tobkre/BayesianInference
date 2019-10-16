# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:40:27 2019

@author: kretz01
"""
import torch

def mse_loss(y_true, y_hat, sigma=0):
    return torch.mean(torch.pow(y_true-y_hat,2))

def nll_loss(y_true, y_hat, y_sig):
    tmp = torch.div(torch.pow(y_true-y_hat,2),2*torch.pow(y_sig,2))
    loss = torch.sum(torch.div(torch.log(torch.pow(y_sig,2)),2) + tmp)
    return loss