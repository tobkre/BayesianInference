# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:57:21 2019

@author: kretz01
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:34:05 2019

@author: kretz01
"""
import copy
import torch
from torch.autograd import Variable
import numpy as np

from utility.lossFunctions.loss import mse_loss, nll_loss
from utility.fitFunctions.fitting import fit_model, fit_model_adversarial
from utility.models.linearModels import hidden1
from playground.ut import training_parameters

#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed(0)

class ensembleTrainer:
    def __init__(self, n_ensembles, normalization='none', loss='mse_loss', adv_training=False):
        if normalization == 'none':
            self.norm = lambda x: x
        elif normalization == 'zerocenter':
            self.norm = lambda x: x - np.mean(x, axis=0)
        
        if loss == 'mse_loss':
            self.criterion = lambda *args: mse_loss(*args)
        elif loss == 'nll_loss':
            self.criterion = lambda *args: nll_loss(*args)
        
        self.N = n_ensembles
        self.Ensemble = []
        
        self.adv_training = adv_training
                
    def fitEnsemble(self, model_class, model_param, x_train, y_train, train_params):
        if self.adv_training:
            fit = lambda *args: fit_model_adversarial(*args)
        else:
            fit = lambda *args: fit_model(*args)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(self.N):
            model = model_class(model_param)
            model = fit(model, x_train, y_train, self.criterion, self.norm, device, train_params)
            self.Ensemble.append(model)
    
    def predict(self, x, device, var='ensemble'):
        XTest = x
        Samples = [self.Ensemble[i](Variable(torch.FloatTensor(XTest)).to(device)) for i in range(self.N)]
        muTest = torch.stack([tup[0] for tup in Samples]).view(self.N,XTest.shape[0]).cpu().data.numpy()
        sigTest = torch.stack([tup[1] for tup in Samples]).view(self.N,XTest.shape[0]).cpu().data.numpy()
        muEnsemble = (1/self.N * np.sum(muTest, axis=0)).squeeze()
        if var == 'ensemble':
            varEnsemble = (1/self.N * np.sum(sigTest**2 + muTest**2-muEnsemble**2, axis=0)).squeeze()        
        else:
            varEnsemble = (np.std(muTest, axis=0)).squeeze()**2
        sigEnsemble = np.sqrt(varEnsemble)
        return muEnsemble, sigEnsemble, muTest, sigTest

if __name__=='__main__':
    import pylab
    from utility.models.linearModels import hidden1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 20
    N_EPOCHS = 750
    LR = 0.01
    train_params = training_parameters(lr=0.01, n_epochs=400, verbose=False)

    epsilon = np.random.normal(loc=0, scale=1, size=(N,1))*3
    x = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    
    y = x**3 + epsilon
    
    x_tst = np.linspace(start=-6., stop=6., num=100)
    y_tst = x_tst**3
    x_tst = np.reshape(x_tst, (-1,1))
    
    ensemble1 = ensembleTrainer(5, loss='nll_loss', adv_training=True)
    ensemble1.fitEnsemble(hidden1, 500, x, y, train_params)
    
#    ensemble1 = ensembleTrainer(5)
#    ensemble1.fitEnsemble(hidden1, 100, x, y, N_EPOCHS, LR, device, verbose=True)
    
    mu_MSE_5_NOAD, sig_MSE_5_NOAD, mus_MSE_5_NOAD, _ = ensemble1.predict(x_tst, device)
    
    pylab.figure()
    pylab.plot(x_tst, y_tst)
    pylab.scatter(x, y, color='red')
    pylab.fill_between(x_tst.squeeze(), mu_MSE_5_NOAD-3*sig_MSE_5_NOAD, mu_MSE_5_NOAD+3*sig_MSE_5_NOAD, color='gray', alpha=0.5)
    pylab.plot(x_tst, mus_MSE_5_NOAD.T)