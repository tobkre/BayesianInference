# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:23:25 2019

@author: kretz01
"""

import torch
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self, x, y, transforms=None):
        self.X = x
        self.Y = y
        self.Transforms = transforms
        
    def __getitem__(self, index):
        data_point = self.X[index, :]
        data_label = self.Y[index, :]
        return data_point, data_label
    
    def __len__(self):
        return len(self.X)