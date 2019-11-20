# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:35:15 2019

@author: kretz01
"""

import copy
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

import pylab

class concreteLearning:
    def __init__(self, model, weight_regularizer=1e-6, N_samples=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_regularizer = weight_regularizer
        self.model = model
        self.p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.N = N_samples
        self.is_trained = False
    
    def loss(self, true, mean, log_var):
        precision = torch.exp(-log_var)
        return torch.mean(torch.sum(precision*(true-mean)**2+log_var,1),0)
    
    def train(self, x_train, y_train, train_params):
        x = Variable(torch.FloatTensor(x_train)).to(self.device)
        y = Variable(torch.FloatTensor(y_train)).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=train_params.lr)
        
        for i in range(train_params.n_epochs):
            mean, log_var, regularization = self.model(x)
            loss = self.loss(y, mean, log_var) + regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.is_trained = True
    
    def __calc_tau_map(self, y_train, y_pred):
        if self.is_trained:
            y_train = torch.FloatTensor(y_train)
            y_pred = torch.FloatTensor(y_pred)
            n = len(y_train)
            return ((1/(self.p + n - 2))*self.loss(y_train, y_pred))**(-1)
        else:
            return 0
        
    def sample_delta_theta(self, x, y):
        reg=0
        n = len(y)
        for param in self.model.parameters():
            reg += torch.sum(torch.pow(param, 2))
        reg = self.weight_regularizer*reg.cpu().data.numpy()
        y_p = self.model(torch.FloatTensor(x)).cpu().data.numpy()
        gamma = (np.matmul(np.transpose(y-y_p), y-y_p)+reg).flatten()[0]
        alpha = (n+self.p)/2
        delta_tau = 10
        exp_tau = self.tau_map.cpu().data.numpy()
#        print(gamma)
        i=0
        while delta_tau > 1e-2 and i<100:
#            print(exp_tau)
            exp_tau_old = exp_tau
            var_delta = (exp_tau)**-1 * (np.diag(np.matmul(self.jac.T, self.jac))+self.weight_regularizer)**-1
            beta = 0.5 * (gamma+np.sum(var_delta*(np.diag(np.matmul(self.jac.T, self.jac))+self.weight_regularizer)))
            exp_tau = alpha/beta
            delta_tau = np.abs(exp_tau-exp_tau_old)
#            print('delta tau: {:.4f}'.format(delta_tau))
            i+=1
        delta_theta = np.random.normal(loc=0, scale=np.sqrt(var_delta), size=(self.N, len(var_delta)))
        return delta_theta
    
    def predict(self, x):
        if self.is_trained:
            self.model.eval()
            y_pred = self.model(torch.FloatTensor(x))[0].cpu().data.numpy()
            return y_pred
        
    def sample_predict(self, x, y, K=200):
        if self.is_trained:
            self.model.eval()
            MC_samples = [self.model(Variable(torch.FloatTensor(x))) for _ in range(K)]
            means = torch.stack([tup[0] for tup in MC_samples]).view(K, x.shape[0]).cpu().data.numpy()
            logvar = torch.stack([tup[1] for tup in MC_samples]).view(K, x.shape[0]).cpu().data.numpy()
            return np.mean(means, 0), np.std(means, 0)

def get_dev(y_t, y_p, y_std, tau):
    l = len(y_p)
    dev = (np.sum(((y_t-y_p)**2-tau**(-1))/y_std**2)-l)**2
    return dev

def make_plot(x, y, x_tst, y_tst, y_tst_pred, y_tst_samples, lc, uc, textstr):
    pylab.figure()
    pylab.scatter(x, y, color='r')
    pylab.plot(x_tst, y_tst)
    pylab.plot(x_tst,y_tst_pred)
#    pylab.plot(x_tst, y_tst_samples)
    pylab.fill_between(x_tst.squeeze(), lc.squeeze(), uc.squeeze(), color='gray', alpha=0.5)
    pylab.xlim([-7,7])
    pylab.ylim([-250,250])
    pylab.text(-6.8, 150., textstr, fontsize=12)
    
if __name__=='__main__':
    from easy_model import hidden1
    from ut import training_parameters
    N = 50
    np.random.seed(1)
    
    lbda = 2.95e-1
    tau = 1.5e+1
    
    epsilon = np.random.normal(loc=0, scale=1, size=(N,1))*3
#    epsilon=0
#    x = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    x = np.linspace(start=-4, stop=4, num=N).reshape((-1,1))
    y = (x)**3 + epsilon
    
    x_tilde = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    y_tilde = x_tilde**3 + epsilon
        
    x_tst = np.linspace(start=-6, stop=6, num=100).reshape((-1,1))
    y_tst = (x_tst)**3
    
    train_params = training_parameters(lr=0.01, n_epochs=2000, verbose=True)
    model = hidden1(100)