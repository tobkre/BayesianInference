# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:52:28 2019

@author: kretz01
"""

import torch
import numpy as np

from ensemble_learning import ensembleTrainer
from utility.models.linearModels import hidden1

if __name__=='__main__': 
    import pylab
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 20
    N_EPOCHS = 750
    LR = 0.01
    
    epsilon = np.random.normal(loc=0, scale=1, size=(N,1))*3
    x = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    y = x**3 + epsilon
    
    x_tst = np.linspace(start=-6., stop=6., num=100)
    y_tst = x_tst**3
    x_tst = np.reshape(x_tst, (-1,1))
    
    ensemble_MSE_5_NOAD = ensembleTrainer(5, loss='mse_loss')
    ensemble_MSE_5_NOAD.fitEnsemble(hidden1, 100, x, y, N_EPOCHS, LR, device)
    
    ensemble_NLL_1_NOAD = ensembleTrainer(1, loss='nll_loss')
    ensemble_NLL_1_NOAD.fitEnsemble(hidden1, 100, x, y, N_EPOCHS, LR, device)
    
    ensemble_NLL_1_AD = ensembleTrainer(1, loss='nll_loss', adv_training=True)
    ensemble_NLL_1_AD.fitEnsemble(hidden1, 100, x, y, N_EPOCHS, LR, device)
    
    ensemble_NLL_5_AD = ensembleTrainer(5, loss='nll_loss', adv_training=True)
    ensemble_NLL_5_AD.fitEnsemble(hidden1, 100, x, y, N_EPOCHS, LR, device)
    
    mu_MSE_5_NOAD, sig_MSE_5_NOAD, mus_MSE_5_NOAD, _ = ensemble_MSE_5_NOAD.predict(x_tst, device, var='mse')
    mu_NLL_1_NOAD, sig_NLL_1_NOAD, mus_NLL_1_NOAD, _ = ensemble_NLL_1_NOAD.predict(x_tst, device)
    mu_NLL_1_AD, sig_NLL_1_AD, mus_NLL_1_AD, _ = ensemble_NLL_1_AD.predict(x_tst, device)
    mu_NLL_5_AD, sig_NLL_5_AD, mus_NLL_5_AD, _ = ensemble_NLL_5_AD.predict(x_tst, device)
       
    _, (ax1, ax2, ax3, ax4) = pylab.subplots(1,4, figsize=(20,3), sharex=True, sharey=True)
    ax1.plot(x_tst, y_tst)
    ax1.scatter(x, y, color='red')
    ax1.fill_between(x_tst.squeeze(), mu_MSE_5_NOAD-3*sig_MSE_5_NOAD, mu_MSE_5_NOAD+3*sig_MSE_5_NOAD, color='gray', alpha=0.5)
#    ax1.plot(x_tst, mus_MSE_5_NOAD.T)
    
    ax2.plot(x_tst, y_tst)
    ax2.scatter(x, y, color='red')
    ax2.fill_between(x_tst.squeeze(), mu_NLL_1_NOAD-3*sig_NLL_1_NOAD, mu_NLL_1_NOAD+3*sig_NLL_1_NOAD, color='gray', alpha=0.5)
#    ax2.plot(x_tst, mus_NLL_1_NOAD.T)
    
    ax3.plot(x_tst, y_tst)
    ax3.scatter(x, y, color='red')
    ax3.fill_between(x_tst.squeeze(), mu_NLL_1_AD-3*sig_NLL_1_AD, mu_NLL_1_AD+3*sig_NLL_1_AD, color='gray', alpha=0.5)
#    ax3.plot(x_tst, mus_NLL_1_AD.T)
#    
    ax4.plot(x_tst, y_tst)
    ax4.scatter(x, y, color='red')
    ax4.fill_between(x_tst.squeeze(), mu_NLL_5_AD-3*sig_NLL_5_AD, mu_NLL_5_AD+3*sig_NLL_5_AD, color='gray', alpha=0.5)
#    ax4.plot(x_tst, mus_NLL_5_AD.T)