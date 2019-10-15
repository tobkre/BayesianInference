# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:34:05 2019

@author: kretz01
"""

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn 

from pdb import set_trace
import numpy as np
import pylab

#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 20
N_TST = 500
N_MC = 100
N_EPOCHS = 40
N_PRINT = 10
M = 5

def my_loss(true, guess, sigma):
    constant = 1e-6
#    return torch.mean(torch.log(sigma)/2 + torch.pow(true-guess,2)/(2*sigma)+constant)
    return torch.mean(torch.pow(true-guess,2))

def my_y(x, epsilon):
    return x**3 + epsilon

class myModel(nn.Module):
    def __init__(self, nb_features, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super(myModel, self).__init__()
        
        self.linear1 = nn.Linear(1, nb_features)
        
        self.linear2_mu = nn.Linear(nb_features, 1)
        self.linear2_sigma = nn.Linear(nb_features, 1)
        
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus(beta=1, threshold=20)
    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        
        mu = self.linear2_mu(x1)    
        sigma = self.softplus(self.linear2_sigma(x1))
        
        return mu, sigma 

def fit_model(n_epoch, X, Y):
    _x = Variable(torch.FloatTensor(X)).to(device)
    _y = Variable(torch.FloatTensor(Y)).to(device)
    model = myModel(100).to(device);
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(n_epoch):
        mu_model, sigma_model = model(_x)
        
        loss = my_loss(_y, mu_model, sigma_model)
        if epoch % N_PRINT == 0:
            print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch,n_epoch,loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

epsilon = np.random.normal(loc=0, scale=9, size=(N,1))
x = np.random.uniform(low=-4., high=4., size=(N,1))
y = my_y(x, epsilon)

ensemble = []

for i in range(M):
    print(i)
    model = fit_model(N_EPOCHS, x, y)
    ensemble.append(model)
    
x_tst = np.linspace(start=-6., stop=6., num=100)
x_tst = np.reshape(x_tst, (-1,1))

p_samples = [ensemble[i](Variable(torch.FloatTensor(x_tst)).to(device)) for i in range(M)]
p_tst = torch.stack([tup[0] for tup in p_samples]).view(M,x_tst.shape[0]).cpu().data.numpy()

pylab.figure()
pylab.plot(x_tst, my_y(x_tst,0))
pylab.scatter(x, y, color='red')
pylab.plot(x_tst, p_tst.T)
