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

from utility.lossFunctions.loss import nll_loss
from utility.fitFunctions.fitting import fit_model, fit_model_adversarial
from utility.models.linearModels import hidden1
from playground.ut import training_parameters

#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed(0)

class ensembleTrainer:
    def __init__(self, n_ensembles, normalization='none', noise='homoscedastic', adv_training=False, weight_regularizer=1e-6):
        if normalization == 'none':
            self.norm = lambda x: x
        elif normalization == 'zerocenter':
            self.norm = lambda x: x - np.mean(x, axis=0)
        
        self.noise = noise
        
        self.N = n_ensembles
        self.Ensemble = []
        
        self.adv_training = adv_training
        self.weight_regularizer = weight_regularizer
                
    def fitEnsemble(self, model_class, model_param, x_train, y_train, train_params):
        if self.adv_training:
            fit = lambda *args: fit_model_adversarial(*args)
        else:
            fit = lambda *args: fit_model(*args)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(self.N):
            model = model_class(model_param, weight_regularizer=self.weight_regularizer, noise=self.noise)
            model = fit(model, x_train, y_train, nll_loss, self.norm, device, train_params)
            self.Ensemble.append(model)
    
    def predict(self, x, device):
        XTest = x
        Samples = [self.Ensemble[i](torch.FloatTensor(XTest).to(device)) for i in range(self.N)]
        muTest = torch.stack([tup[0] for tup in Samples]).view(self.N,XTest.shape[0]).cpu().data.numpy()
        sigTest = torch.stack([tup[1] for tup in Samples]).view(self.N,XTest.shape[0]).cpu().data.numpy()
        muEnsemble = (1/self.N * np.sum(muTest, axis=0)).squeeze()
#        varEnsemble = (1/self.N * np.sum(sigTest**2 + muTest**2-muEnsemble**2, axis=0)).squeeze()        
        varEnsemble = (1/self.N*(np.sum(sigTest**2 + muTest**2,0))-muEnsemble**2).squeeze()        
        sigEnsemble = np.sqrt(varEnsemble)
        return muEnsemble, sigEnsemble, muTest, sigTest

if __name__=='__main__':
    import pylab
    from utility.models.linearModels import hidden1
    np.random.seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 50

    train_params = training_parameters(lr=0.1, n_epochs=100, verbose=False)

    epsilon = np.random.normal(loc=0, scale=1, size=(N,1))*3
    x = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    
    y = x**3 + epsilon
    
    x_tst = np.linspace(start=-6., stop=6., num=100).reshape((-1,1))
    y_tst = x_tst**3
    
    ensemble1 = ensembleTrainer(5, weight_regularizer=1.95e-2, noise='heteroscedastic', adv_training=False)
    ensemble1.fitEnsemble(hidden1, 100, x, y, train_params)
    
#    ensemble1 = ensembleTrainer(5)
#    ensemble1.fitEnsemble(hidden1, 100, x, y, N_EPOCHS, LR, device, verbose=True)
    
    mu_MSE_5_NOAD, sig_MSE_5_NOAD, mus_MSE_5_NOAD, _ = ensemble1.predict(x_tst, device)
    
    pylab.figure()
    pylab.plot(x_tst, y_tst)
    pylab.scatter(x, y)
    pylab.errorbar(x_tst, mu_MSE_5_NOAD, sig_MSE_5_NOAD)
    pylab.xlim([-6.1,6.1])
    pylab.ylim([-6.1**3,6.1**3])
#    pylab.plot(x_tst, mus_MSE_5_NOAD.T)