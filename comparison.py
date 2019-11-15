# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:16:26 2019

@author: kretz01
"""
import torch
import numpy as np
import pylab

from playground.ut import training_parameters
from playground.elster_learning import elsterLearning
from playground.easy_model import hidden1
from ensemble_learning import ensembleTrainer

# define model function
def f(x, sigma=0):
    N = len(x)
    epsilon = sigma * np.random.normal(loc=0, scale=1, size=(N,1))
    return x**3 + epsilon

def make_plot(x_train, y_train, x_tst, y_tst, y_tst_pred, y_tst_samples, lc, uc):
    pylab.figure()
    pylab.scatter(x_train, y_train, color='r')
    pylab.plot(x_tst, y_tst)
    pylab.plot(x_tst,y_tst_pred)
    pylab.fill_between(x_tst.squeeze(), lc.squeeze(), uc.squeeze(), color='gray', alpha=0.5)
    pylab.xlim([-7,7])
    pylab.ylim([-250,250])

if __name__=='__main__':
    # define parameters
    N = 50
    train_params = training_parameters(lr=0.01, n_epochs=2000, verbose=True)
    
    lbda = 2.95e-1
    tau = 1.5e+1
    
    # define train data and test data
    x_train = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    x_tilde = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    x_tst = np.linspace(start=-6, stop=6, num=100).reshape((-1,1))
    
    y_train = f(x_train, 3)
    y_tilde = f(x_tilde, 3)
    y_tst = f(x_tst)
    
    # define model structure
    model = hidden1(100)
    
    # make 
    par = elsterLearning(model, weight_regularizer=lbda)
    par.train(x_train, y_train, train_params)
    
    y_tst_pred= par.predict(x_tst)
    y_tst_samples, y_tst_std, U_theta = par.sample_predict(x_tst, y_tst, tau=tau)
    lc = y_tst_samples - y_tst_std
    uc = y_tst_samples + y_tst_std
    
    ens = ensembleTrainer(5, loss='nll_loss', adv_training=True)
    ens.fitEnsemble(hidden1, 100, x_train, y_train, train_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu_MSE_5_NOAD, sig_MSE_5_NOAD, mus_MSE_5_NOAD, _ = ens.predict(x_tst, device)
    
    make_plot(x_train, y_train, x_tst, y_tst, y_tst_pred, y_tst_samples, lc, uc)