# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:40:27 2019

@author: kretz01
"""
import torch


def nll_loss(y_true, y_hat, logvar):
    ls = 0.5 * torch.exp(-logvar) * torch.pow(y_true-y_hat,2)
    return (ls + 0.5*logvar).sum()