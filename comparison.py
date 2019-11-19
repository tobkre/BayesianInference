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
from playground.concreteLearning import concreteLearning
from playground.easy_model import hidden1
from ensemble_learning import ensembleTrainer
from utility.models.linearModels import hidden1 as h1
from utility.models.concreteModels import myNet


# define model function
def f(x, sigma=0):
    N = len(x)
    epsilon = sigma * np.random.normal(loc=0, scale=1, size=(N,1))
    return x**3 + epsilon

def make_plot(x_train, y_train, x_tst, y_tst, y_tst_pred, y_tst_samples, y_tst_sig):
    lc = y_tst_pred.squeeze() - y_tst_sig.squeeze()
    uc = y_tst_pred.squeeze() + y_tst_sig.squeeze()
    _, (ax1, ax2) = pylab.subplots(1,2, figsize=(12,5), sharex=True, sharey=False)
    ax1.scatter(x_train, y_train, color='r')
    ax1.plot(x_tst, y_tst)
    ax1.plot(x_tst,y_tst_pred)
    ax1.fill_between(x_tst.squeeze(), lc.squeeze(), uc.squeeze(), color='gray', alpha=0.5)
    ax1.set_xlim(-7,7)
    ax1.set_ylim(-250,250)
    
    diff = np.abs(y_tst.squeeze()-y_tst_pred.squeeze())
    ax2.plot(x_tst, diff)
    ax2.fill_between(x_tst.squeeze(), (diff.squeeze()-y_tst_sig.squeeze()).squeeze(), (diff.squeeze()+y_tst_sig.squeeze()).squeeze(), color='gray', alpha=0.5)
    xz = np.linspace(-8,8, 100)
    ax2.plot(xz, np.zeros_like(xz))

if __name__=='__main__':
    # define parameters
    N = 50
    n_neurons = 50
    
    train_params = training_parameters(lr=0.01, n_epochs=400, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lbda = 1.95e-1
    tau = 0.35e+1
    
    # define train data and test data
    x_train = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    x_tilde = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    x_tst = np.linspace(start=-6, stop=6, num=100).reshape((-1,1))
    
    y_train = f(x_train, 3)
    y_tilde = f(x_tilde, 3)
    y_tst = f(x_tst)
    
    # define model structure
    model = hidden1(n_neurons)
    
    # make 
    par = elsterLearning(model, weight_regularizer=lbda)
    par.train(x_train, y_train, train_params)
    
    y_tst_pred= par.predict(x_tst)
    y_tst_samples, y_tst_std, U_theta = par.sample_predict1(x_tst, y_tst, tau=tau)
    
    make_plot(x_train, y_train, x_tst, y_tst, y_tst_pred, y_tst_samples, y_tst_std)
    
    ens = ensembleTrainer(20, loss='nll_loss', adv_training=True)
    ens.fitEnsemble(h1, n_neurons, x_train, y_train, train_params)
    mu_5, sig_5, mus_5, _ = ens.predict(x_tst, device)
    
    make_plot(x_train, y_train, x_tst, y_tst, mu_5, mu_5, sig_5)
    
    model_conc = myNet(n_neurons, p_init=5.4e-01, train_p=False)
    conc = concreteLearning(model_conc)
    conc.train(x_train, y_train, train_params)
    
    yc_pred = conc.predict(x_tst)
    yc_samples, yc_std = conc.sample_predict(x_tst, y_tst)
    
    make_plot(x_train, y_train, x_tst, y_tst, yc_samples, yc_samples, yc_std)

    for module in model_conc.modules():
        if hasattr(module, 'p_logit'):
            print('drop rate = {:1.1e}'.format(torch.sigmoid(module.p_logit).cpu().data.numpy().flatten()[0]))
    