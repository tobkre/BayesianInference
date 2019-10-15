# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:45:15 2019

@author: kretz01
"""
import torch
from torch.autograd import Variable
from torch import optim

from pdb import set_trace
import numpy as np
import pylab

from model import myNet
from data import myData

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 10000

N_EPOCHS = 1000

S_MAX = 0.25

training_data = myData(N=N, s_max=S_MAX)
X, Y, b0s, b1s, ss = training_data.get_data()

def print_dropout(model):
    print('')
    print('Model parameters:')
    for module in model.modules():
        if hasattr(module, 'p_logit'):
            print('drop rate = {:1.1e}'.format(torch.sigmoid(module.p_logit).cpu().data.numpy().flatten()[0]))

def plot_pred_vs_gt(gt, pred_lsqm, pred_model, param, title=''):
    _, (ax1, ax2) = pylab.subplots(1,2, sharex=True, sharey=True)
    ax1.scatter(gt, pred_lsqm, alpha=0.2)
#    ax1.xlabel('ground_truth '+param)
#    ax1.ylabel('lsqm prediction '+param)
    ax2.scatter(gt, pred_model, alpha=0.2)
#    ax2.xlabel('ground_truth '+param)
#    ax2.ylabel('lsqm prediction '+param)
    pylab.title(title)

def f(X, b0, b1):
    return b0 + b1 * X

def calc_cov(b0, b1):
    b0b0 = np.mean((b0-np.mean(b0))*(b0-np.mean(b0)))
    b0b1 = np.mean((b0-np.mean(b0))*(b1-np.mean(b1)))
    b1b0 = np.mean((b1-np.mean(b1))*(b0-np.mean(b0)))
    b1b1 = np.mean((b1-np.mean(b1))*(b1-np.mean(b1)))
    return np.array([[b0b0, b0b1],[b1b0, b1b1]])

def myloss(true, x, b0, b1):
    guess = torch.mul(x, b1) + b0
    return torch.mean(torch.pow(true-guess,2))

def fit_model(n_epoch, X, Y):
    _x = Variable(torch.FloatTensor(X)).to(device)
    _y = Variable(torch.FloatTensor(Y)).to(device)
    model = myNet(1024).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        b0_model, b1_model, reg = model(_y)
        
        loss = myloss(_y, _x, b0_model, b1_model) + reg 
#        loss = myloss(_y, _x, b0_model, b1_model) 
        if epoch % 100 == 0:
#            print(torch.sigmoid(model.conc_drop_b0.p_logit))
#            print(torch.sigmoid(model.conc_drop2.p_logit))
            print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch,n_epoch,loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
    
model = fit_model(N_EPOCHS, X, Y)
torch.save(model, 'testmodel_{:.3f}.pkl'.format(S_MAX))