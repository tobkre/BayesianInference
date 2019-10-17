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
    
    N_REP = 20
    
    mu_MSE_5_NOAD = np.zeros((100,N_REP))
    sig_MSE_5_NOAD = np.zeros((100,N_REP))
    mu_NLL_1_NOAD = np.zeros((100,N_REP))
    sig_NLL_1_NOAD = np.zeros((100,N_REP))
    mu_NLL_1_AD = np.zeros((100,N_REP))
    sig_NLL_1_AD = np.zeros((100,N_REP))
    mus_NLL_1_AD = np.zeros((100,N_REP))
    mu_NLL_5_AD = np.zeros((100,N_REP))
    sig_NLL_5_AD = np.zeros((100,N_REP))
    mus_NLL_5_AD = np.zeros((100,N_REP))
    
    for i in range(N_REP):
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
    
        mu_MSE_5_NOAD[:,i], sig_MSE_5_NOAD[:,i], _, _ = ensemble_MSE_5_NOAD.predict(x_tst, device, var='mse')
        mu_NLL_1_NOAD[:,i], sig_NLL_1_NOAD[:,i], _, _ = ensemble_NLL_1_NOAD.predict(x_tst, device)
        mu_NLL_1_AD[:,i], sig_NLL_1_AD[:,i], _, _ = ensemble_NLL_1_AD.predict(x_tst, device)
        mu_NLL_5_AD[:,i], sig_NLL_5_AD[:,i], _, _ = ensemble_NLL_5_AD.predict(x_tst, device)
    
    lc_MSE_5_NOAD = np.mean(mu_MSE_5_NOAD, axis=1) - 3*np.mean(sig_MSE_5_NOAD, axis=1)
    uc_MSE_5_NOAD = np.mean(mu_MSE_5_NOAD, axis=1) + 3*np.mean(sig_MSE_5_NOAD, axis=1)
    
    lc_NLL_1_NOAD = np.mean(mu_NLL_1_NOAD, axis=1) - 3*np.mean(sig_NLL_1_NOAD, axis=1)
    uc_NLL_1_NOAD = np.mean(mu_NLL_1_NOAD, axis=1) + 3*np.mean(sig_NLL_1_NOAD, axis=1)
    
    lc_NLL_1_AD = np.mean(mu_NLL_1_AD, axis=1) - 3*np.mean(sig_NLL_1_AD, axis=1)
    uc_NLL_1_AD = np.mean(mu_NLL_1_AD, axis=1) + 3*np.mean(sig_NLL_1_AD, axis=1)
    
    lc_NLL_5_AD = np.mean(mu_NLL_5_AD, axis=1) - 3*np.mean(sig_NLL_5_AD, axis=1)
    uc_NLL_5_AD = np.mean(mu_NLL_5_AD, axis=1) + 3*np.mean(sig_NLL_5_AD, axis=1)
    
    _, (ax1, ax2, ax3, ax4) = pylab.subplots(1,4, figsize=(18,3), sharex=True, sharey=True)
    ax1.plot(x_tst, y_tst)
    ax1.scatter(x, y, color='red')
    ax1.fill_between(x_tst.squeeze(), lc_MSE_5_NOAD, uc_MSE_5_NOAD, color='gray', alpha=0.5)
    ax1.set_title('MSE, M=5, No Adv Ex')
    
    ax2.plot(x_tst, y_tst)
    ax2.scatter(x, y, color='red')
    ax2.fill_between(x_tst.squeeze(), lc_NLL_1_NOAD, uc_NLL_1_NOAD, color='gray', alpha=0.5)
    ax2.set_title('NLL, M=1, No Adv Ex')
    
    ax3.plot(x_tst, y_tst)
    ax3.scatter(x, y, color='red')
    ax3.fill_between(x_tst.squeeze(), lc_NLL_1_AD, uc_NLL_1_AD, color='gray', alpha=0.5)
    ax3.set_title('NLL, M=1, Adv Ex')
    
    ax4.plot(x_tst, y_tst)
    ax4.scatter(x, y, color='red')
    ax4.fill_between(x_tst.squeeze(), lc_NLL_5_AD, uc_NLL_5_AD, color='gray', alpha=0.5)
    ax4.set_title('NLL, M=5, Adv Ex')
#    pylab.savefig('LakschmiFigure1.jpg', dpi=300)
