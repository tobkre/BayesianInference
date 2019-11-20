# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:31:46 2019

@author: kretz01
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:30:22 2019

@author: kretz01
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

from ut import training_parameters

def f(x, sigma=0):
    N = len(x)
    epsilon = sigma * np.random.normal(loc=0, scale=1, size=(N,1))
    return x**3 + epsilon

class test_model(nn.Module):
    def __init__(self, n_hidden, weight_regularizer=1e-4, sigma_init=12):
        super(test_model, self).__init__()
        self.weight_regularizer = weight_regularizer
        
        self.linear1 = nn.Linear(1, n_hidden) 
        self.mu = nn.Linear(n_hidden, 1)
        self.var = nn.Linear(n_hidden, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        mu = self.mu(x1)
        sig = self.var(x1)
        return mu, sig

def my_loss(yt, yp, sigma):
    ls = torch.div((torch.pow(yt-yp,2)), 2*torch.pow(sigma,2))
    return (ls + 0.5*torch.log(torch.pow(sigma,2))).sum()

def train(model, x_train, y_train, train_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _x = Variable(torch.FloatTensor(x_train)).to(device)
    _y = Variable(torch.FloatTensor(y_train)).to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_params.lr)
    for epoch in range(train_params.n_epochs):
        y_p, sig_p = model(_x)
        loss = my_loss(_y, y_p, sig_p)
#        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        
        
if __name__=='__main__':
    import pylab
    N = 50
    t_par = training_parameters(lr=0.001, n_epochs=20000, verbose=False)
    
    x_train = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    y_train = f(x_train, 1)
    
    model = test_model(100)
    
    train(model, x_train, y_train, t_par)
    
    model.eval()
    y_p, sig_p = model(torch.FloatTensor(x_train))
    lc = (y_p - sig_p).data.cpu().numpy().squeeze()
    uc = (y_p + sig_p).data.cpu().numpy().squeeze()
    
    print(np.mean(np.sqrt((sig_p**2).data.cpu().numpy())))
    pylab.figure()
    pylab.plot(x_train, x_train**3)
    pylab.plot(x_train, y_p.data.cpu().numpy())
    pylab.fill_between(x_train.squeeze(), lc, uc, alpha=0.5)
    
    
    