# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:48:49 2019

@author: kretz01
"""

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
import copy

import pylab

from easy_model import hidden1
from ut import training_parameters

def f(x, sigma=0):
    N = len(x)
    epsilon = sigma * np.random.normal(loc=0, scale=1, size=(N,1))
    return x**3 + epsilon

def rmse_loss(y_t, y_p):
    return torch.sum(torch.pow((y_t-y_p),2))

def train(model, x_train, y_train, train_par):
    model.train()
    optimizer = Adam(model.parameters(), lr=train_par.lr)
    _x = Variable(torch.FloatTensor(x_train))
    _y = Variable(torch.FloatTensor(y_train))
    for epoch in range(train_par.n_epochs):
        y_p = model(_x)
        loss = rmse_loss(_y, y_p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if train_par.verbose and (epoch%train_par.verbose_frequency==0 or epoch==train_par.n_epochs-1):
            print('Epoch {:d}/{:d} Loss={:.4f}'.format(epoch+1, train_par.n_epochs, loss))
    return model

def predict(models, x_tst):
    M = len(models)
    y_pred = np.zeros((M, len(x_tst)))
    for i in range(M):
        model = models[i]
        model.eval()
        y_pred[i,:] = model(torch.FloatTensor(x_tst)).data.cpu().numpy().squeeze()
    return y_pred

def get_dist(models):
    M = len(models)
    thetas = []
    for i in range(M):
        model = models[i]
        theta = np.empty(0)
        for param in model.parameters():
            theta = np.concatenate((theta, param.flatten().cpu().data.numpy()))
        thetas.append(theta)
    return np.array(thetas)

if __name__=='__main__':
    N = 200
    M = 200
    frac = 0.25
    
    train_par = training_parameters(lr=0.05, n_epochs=2000, verbose=True, verbose_frequency=1000)
    
    x_train = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    x_tst = np.linspace(start=-6, stop=6, num=150).reshape((-1,1))
    
    y_train = f(x_train, 3)
    y_tst = f(x_tst)
    
    model = hidden1(100)
    
    lcs = []
    ucs = []
    yms = []
    trans_par = training_parameters(lr=0.025, n_epochs=200, verbose=False)
    for _ in range(2):
        models = []
        x_train = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
        y_train = f(x_train, 3)
        par_model = train(copy.deepcopy(model), x_train, y_train, train_par)
        for i in range(M):
            n_boot = int(frac*N)
            i_boot = np.random.choice(N, n_boot, replace=True)
            x_train_boot = x_train[i_boot,:]
            y_train_boot = y_train[i_boot,:]
            b_model = train(par_model, x_train_boot, y_train_boot, trans_par)
            models.append(copy.deepcopy(b_model))
        
        
        y_pred = predict(models, x_tst)
    
        lcs.append((y_pred.mean(0) - y_pred.std(0)).squeeze())
        ucs.append((y_pred.mean(0) + y_pred.std(0)).squeeze())
        yms.append(y_pred.mean(0).squeeze())
    
    thetas = get_dist(models)
    
    pylab.figure()
    pylab.plot(x_tst, y_tst)
    pylab.scatter(x_train, y_train, color='r')
#    pylab.plot(x_tst, y_pred.T)
    pylab.plot(x_tst, np.array(yms).mean(0))
    pylab.fill_between(x_tst.squeeze(), np.array(lcs).mean(0).squeeze(), np.array(ucs).mean(0).squeeze(), color='gray', alpha=0.5)
    pylab.xlim([-6.5,6.5])
    pylab.ylim([-6.5**3,6.5**3])
    
    pylab.figure()
    _, (ax1, ax2, ax3, ax4, ax5, ax6) = pylab.subplots(2,3, sharex=True, sharey=True)
    ind = np.argsort(thetas.std(0))
    ax1.hist(thetas[:,ind[-1)]], bins=200, range=(thetas.min(), thetas.max()))
        
    