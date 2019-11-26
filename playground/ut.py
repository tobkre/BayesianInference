# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:26:46 2019

@author: kretz01
"""

class training_parameters:
    def __init__(self, lr=0.01, n_epochs=750, verbose=False, verbose_frequency=400):
        self.lr = lr
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.verbose_frequency = verbose_frequency
        