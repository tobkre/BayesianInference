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
from torch import optim
import torch.nn as nn 

from pdb import set_trace
import numpy as np
import pylab

#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed(0)

class myModel(nn.Module):
    def __init__(self, nb_features, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super(myModel, self).__init__()
        
        self.linear1 = nn.Linear(1, nb_features)
        
        self.linear2_mu = nn.Linear(nb_features, 1)
        self.linear2_sigma = nn.Linear(nb_features, 1)
        
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus(beta=1, threshold=20)
    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        
        mu = self.linear2_mu(x1)    
        sigma = self.softplus(self.linear2_sigma(x1)) + 1e-6
#        sigma = torch.log(1+torch.exp(self.linear2_sigma(x1))) + 1e-6
#        sigma = self.linear2_sigma(x1)
        
        return mu, sigma

def fit_model(criterion, n_epoch, X, Y, n_hid, device, verbose):
    _x = Variable(torch.FloatTensor(X)).to(device)
    _y = Variable(torch.FloatTensor(Y)).to(device)
    model = myModel(n_hid).to(device);
    optimizer = optim.Adam(model.parameters(), lr=0.1)#, weight_decay=1e-3)
    for epoch in range(n_epoch):
        mu, sig = model(_x)
        sum_of_square=0
        for param in model.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        loss = criterion(_y, mu, sig) + 1e-3*sum_of_square
        if torch.isinf(loss):
            print("Stop")
        if verbose and (epoch == 0 or epoch % 50 == 0 or epoch == n_epoch-1):
            print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch+1,n_epoch,loss))
#            print("Example: true: {:.3f} guess:{:.3f} +- {:.3f}".format(_y[0].cpu().data.numpy()[0], mu[0].cpu().data.numpy()[0], var[0].cpu().data.numpy()[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def fit_model_adversarial(criterion, n_epoch, X, Y, n_hid, device, verbose, alpha=0.5):
    epsilon = 0.01*8
    model = myModel(n_hid).to(device);
    _x = Variable(torch.FloatTensor(X)).to(device)
    _y = Variable(torch.FloatTensor(Y)).to(device)
    _x_adv = perturb_x(X, Y, epsilon, model, criterion).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(n_epoch):
        mu, var = model(_x)
        mu_ad, var_ad = model(_x_adv)
        loss = alpha * criterion(_y, mu, var) + (1-alpha) * criterion(_y, mu_ad, var_ad)
        if verbose and (epoch == 0 or epoch % 10 == 0 or epoch==n_epoch-1):
            print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch+1,n_epoch,loss))
            print("Example: true: {:.3f} guess:{:.3f} +- {:.3f}".format(_y[0].cpu().data.numpy()[0], mu[0].cpu().data.numpy()[0], np.sqrt(var[0].cpu().data.numpy())[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def mse_loss(y_true, y_hat, sigma=0):
    return torch.mean(torch.pow(y_true-y_hat,2))

def nll_loss(y_true, y_hat, y_sig):
    tmp = torch.div(torch.pow(y_true-y_hat,2),2*torch.pow(y_sig,2))
    loss = torch.sum(torch.div(torch.log(torch.pow(y_sig,2)),2) + tmp)
#    y_sig = torch.log(1+torch.exp(y_sig)) + 1e-6
    return loss

def perturb_x(x_nat, y, epsilon, model, criterion):
    model_cp = copy.deepcopy(model)
    x = np.copy(x_nat)
    x_var = Variable(torch.FloatTensor(x), requires_grad=True)
    y_var = Variable(torch.FloatTensor(y))
    
    scores, sigmas = model_cp(x_var)
    loss = criterion(y_var, scores, sigmas)
    loss.backward()
    grad_sign = x_var.grad.data.cpu().sign().numpy()
    x_adv = x + epsilon*grad_sign
    return Variable(torch.FloatTensor(x_adv))

def train_ensemble(n_ens, n_hid, x, y, n_ep, loss='mse_loss', adv_training=False, verbose=False):
    if adv_training:
        fit = lambda *args: fit_model_adversarial(*args)
    else:
        fit = lambda *args: fit_model(*args)
        
    if loss == 'mse_loss':
        criterion = lambda *args: mse_loss(*args)
    elif loss == 'nll_loss':
        criterion = lambda *args: nll_loss(*args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = []
    for i in range(n_ens):
        model = fit(criterion, n_ep, x, y, n_hid, device, verbose)
        ensemble.append(model)
    return ensemble


if __name__=='__main__':
    N = 20
    N_EPOCHS = 1000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def my_y(x, epsilon):
        return x**3 + epsilon
    
    def pred(ensemble, x_tst, loss='nll'):
        M = len(ensemble)
        p_samples = [ensemble[i](Variable(torch.FloatTensor(x_tst)).to(device)) for i in range(M)]
        mu_tst = torch.stack([tup[0] for tup in p_samples]).view(M,x_tst.shape[0]).cpu().data.numpy()
        sig_tst = torch.stack([tup[1] for tup in p_samples]).view(M,x_tst.shape[0]).cpu().data.numpy()
        mu_star = (1/M * np.sum(mu_tst, axis=0)).squeeze()
        if loss == 'nll':
            var_star = (1/M * np.sum(sig_tst**2 + mu_tst**2-mu_star**2, axis=0)).squeeze()        
        else:
            var_star = (np.std(mu_tst, axis=0)).squeeze()**2
#        print(var_star)
#        return mu_star, np.sqrt(var_star), mu_tst
        return mu_tst, sig_tst, mu_star, np.sqrt(var_star)

    epsilon = np.random.normal(loc=0, scale=1, size=(N,1))*3
    x = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
    y = my_y(x, epsilon)
    print('MSE, 5:')
    ensemble_MSE_5_NOAD = train_ensemble(5, 100, x, y, N_EPOCHS, loss='mse_loss')
    print()
    print('NLL, 1:')
    ensemble_NLL_1_NOAD = train_ensemble(1, 100, x, y, N_EPOCHS, loss='nll_loss', verbose=True)
    print()
    print('NLL, 1:')
    ensemble_NLL_1_AD = train_ensemble(1, 100, x, y, N_EPOCHS, loss='nll_loss', adv_training=True)
    print()
    print('NLL, 5:')
    ensemble_NLL_5_AD = train_ensemble(5, 100, x, y, N_EPOCHS, loss='nll_loss', adv_training=True)

    x_tst = np.linspace(start=-6., stop=6., num=100)
    x_tst = np.reshape(x_tst, (-1,1))
    tmp = ensemble_NLL_1_NOAD[0](Variable(torch.FloatTensor(x_tst)).to(device))
    out = tmp[1].cpu().data.numpy()
    
    print(np.squeeze(out))
#    mu_p_MSE_5_NOAD, sig_p_MSE_5_NOAD, _ = pred(ensemble_MSE_5_NOAD, x_tst, loss='mse')
#    mu_p_NLL_1_NOAD, sig_p_NLL_1_NOAD, mu = pred(ensemble_NLL_1_NOAD, x_tst)
#    mu_p_NLL_1_AD, sig_p_NLL_1_AD, _ = pred(ensemble_NLL_1_AD, x_tst)
#    mu_p_NLL_5_AD, sig_p_NLL_5_AD, _ = pred(ensemble_NLL_5_AD, x_tst)
    mus_MSE_5_NOAD, sig_MSE_5_NOAD, mu_MSE_5_NOAD, std_MSE_5_NOAD = pred(ensemble_MSE_5_NOAD, x_tst, loss='mse')
    mus_NLL_1_NOAD, sig_NLL_1_NOAD, mu_NLL_1_NOAD, std_NLL_1_NOAD = pred(ensemble_NLL_1_NOAD, x_tst)
    mus_NLL_1_AD, sig_NLL_1_AD, mu_NLL_1_AD, std_NLL_1_AD = pred(ensemble_NLL_1_AD, x_tst)
    mus_NLL_5_AD, sig_NLL_5_AD, mu_NLL_5_AD, std_NLL_5_AD = pred(ensemble_NLL_5_AD, x_tst)
    
#    ax1.scatter(gt, pred_lsqm, alpha=0.2)
#    ax1.set_title(r'${}$ lsqm'.format(title))
#    ax1.set_xlabel('ground truth')
#    ax1.set_ylabel('prediction')
#    ax2.scatter(gt, pred_model, alpha=0.2)
#    ax2.set_title(r'${}$ neural net'.format(title))
#    ax2.set_xlabel('ground truth')
#    ax2.set_ylabel('prediction')
    
    _, (ax1, ax2, ax3, ax4) = pylab.subplots(1,4, figsize=(20,3), sharex=True, sharey=True)
    ax1.plot(x_tst, my_y(x_tst,0))
    ax1.scatter(x, y, color='red')
    ax1.fill_between(x_tst.squeeze(), mu_MSE_5_NOAD-3*std_MSE_5_NOAD, mu_MSE_5_NOAD+3*std_MSE_5_NOAD, color='gray', alpha=0.5)
    ax1.plot(x_tst, mus_MSE_5_NOAD.T)
    
    ax2.plot(x_tst, my_y(x_tst,0))
    ax2.scatter(x, y, color='red')
    ax2.fill_between(x_tst.squeeze(), mu_NLL_1_NOAD-3*std_NLL_1_NOAD, mu_NLL_1_NOAD+3*std_NLL_1_NOAD, color='gray', alpha=0.5)
    ax2.plot(x_tst, mus_NLL_1_NOAD.T)
    
    ax3.plot(x_tst, my_y(x_tst,0))
    ax3.scatter(x, y, color='red')
    ax3.fill_between(x_tst.squeeze(), mu_NLL_1_AD-3*std_NLL_1_AD, mu_NLL_1_AD+3*std_NLL_1_AD, color='gray', alpha=0.5)
    ax3.plot(x_tst, mus_NLL_1_AD.T)
#    
    ax4.plot(x_tst, my_y(x_tst,0))
    ax4.scatter(x, y, color='red')
    ax4.fill_between(x_tst.squeeze(), mu_NLL_5_AD-3*std_NLL_5_AD, mu_NLL_5_AD+3*std_NLL_5_AD, color='gray', alpha=0.5)
    ax4.plot(x_tst, mus_NLL_5_AD.T)