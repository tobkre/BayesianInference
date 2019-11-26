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
    def __init__(self, n_hidden, weight_regularizer, noise='homoscedastic'):
        super(test_model, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.noise = noise
        
        self.linear1 = nn.Linear(1, n_hidden) 
        self.mu = nn.Linear(n_hidden, 1)
        
        if noise=='homoscedastic':
            self.logvar = nn.Parameter(torch.FloatTensor([1e-1]))
        elif noise=='heteroscedastic':
            self.logvar = nn.Linear(n_hidden, 1, bias=False)
            
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
            reg += self._regularize(self.logvar)
        else:
            logvar = self.logvar * torch.ones_like(x)
        return mu, logvar, self.weight_regularizer*reg
    
    def _regularize(self, layer):
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square

def my_loss(yt, yp, logvar):
    ls = 0.5 * torch.exp(-logvar) * torch.pow(yt-yp,2)
    return (ls + 0.5*logvar).sum()

def train(model, x_train, y_train, train_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _x = Variable(torch.FloatTensor(x_train)).to(device)
    _y = Variable(torch.FloatTensor(y_train)).to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_params.lr)
    for epoch in range(train_params.n_epochs):
        mu, logvar, reg = model(_x)
        loss = my_loss(_y, mu, logvar)+reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if train_params.verbose and (epoch == 0 or epoch % 50 == 0 or epoch == train_params.n_epochs-1):
                print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch+1,train_params.n_epochs,loss))
#        print(loss)
        
        
if __name__=='__main__':
    import pylab
    N = 50
    M = 5
    t_par = training_parameters(lr=0.05, n_epochs=2000, verbose=True)
    
    x_train = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    x_tilde = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    y_train = f(x_train, 3)
    y_tilde = f(x_tilde, 3)
    
    x_tst = np.linspace(-6,6,100).reshape((-1,1))
    y_tst = f(x_tst)
    
    tst = np.linspace(0.5,7,100)
    tmp = np.array([my_loss(torch.Tensor(x_train)**3, torch.Tensor(y_train), torch.log(torch.Tensor([el])**2)).data.cpu().numpy() for el in tst])
    print(tst[np.argmin(tmp)])
    
    res_mu = np.empty((M, len(y_tst)))
    res_sig = np.empty((M, len(y_tst)))
    
    for i in range(M):
        model = test_model(50, weight_regularizer=1.95e-2, noise='no')
        train(model, x_train, y_train, t_par)
#        print(np.sqrt(torch.exp(model.logvar).cpu().data.numpy()))
        model.eval()
        y_p, logvar_p,_ = model(torch.FloatTensor(x_tst))
        lc = (y_p - torch.sqrt(torch.exp(logvar_p))).data.cpu().numpy().squeeze()
        uc = (y_p + torch.sqrt(torch.exp(logvar_p))).data.cpu().numpy().squeeze()

        y_tilde_p, logvar_tilde_p, _ = model(torch.FloatTensor(x_tilde))
#        pylab.figure()
#        pylab.plot(tst, tmp)
    
#        print(sigma)
#        print()
#        pylab.figure()
#        pylab.plot(x_train, x_train**3)
#        pylab.plot(x_train, y_p.data.cpu().numpy())
#        pylab.fill_between(x_train.squeeze(), lc, uc, alpha=0.5)
        res_mu[i,:] = y_p.data.cpu().numpy().squeeze()
        res_sig[i, :] = torch.sqrt(torch.exp(logvar_p)).data.cpu().numpy().squeeze()
    
    var = 1/M*np.sum(res_mu**2, 0)-(1/M*np.sum(res_mu, 0))**2+1/M*np.sum(res_sig**2, 0)
    lc = (np.mean(res_mu, 0) - 3).squeeze()
    uc = (np.mean(res_mu, 0) + 3).squeeze()
    
    pylab.figure()
    pylab.plot(x_tst, y_tst)
    pylab.fill_between(x_tst.squeeze(), (x_tst**3-3).squeeze(), (x_tst**3+3).squeeze(), alpha=0.5)
    pylab.scatter(x_train, y_train)
    pylab.errorbar(x_tst, np.mean(res_mu, 0), np.sqrt(var))
#    pylab.fill_between(x_tst.squeeze(), lc, uc, alpha=0.5)
    pylab.xlim([-6,6])
#    pylab.ylim([-6.1**3,6.1**3])
    pylab.ylim([-200,200])