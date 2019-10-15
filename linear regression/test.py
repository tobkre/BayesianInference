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
    
def test_covariances(model, S_MAX):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_TST = 500
    N_MC = 100
    
    model = model.to(device)
    model.eval()
    
    tst_data_1 = myData(N=N_TST, s_max=S_MAX)
    X_tst_1, Y_tst_1, b0s_tst_1, b1s_tst_1, ss_tst_1 = tst_data_1.get_data()
    
    tst_data_2 = myData(N=N_TST, s_max=S_MAX)
    X_tst, Y_tst, b0s_tst, b1s_tst, ss_tst = tst_data_2.get_data()
    
    print_dropout(model)
    
    # test model on tst_data_1
    b0s_pred_1, b1s_pred_1, _ = model(Variable(torch.FloatTensor(Y_tst_1)).to(device))
    b0s_pred_1 = b0s_pred_1.cpu().data.numpy().flatten()
    b1s_pred_1 = b1s_pred_1.cpu().data.numpy().flatten()
    # calculate covariance on prediction
    pred_1_cov = calc_cov(b0s_pred_1-b0s_tst_1, b1s_pred_1-b1s_tst_1)
    print(pred_1_cov)
    
     
    MC_SAMPLES = [model(Variable(torch.FloatTensor(Y_tst)).to(device)) for _ in range(N_MC)]
    b0_model = torch.stack([tup[0] for tup in MC_SAMPLES]).view(N_MC,Y_tst.shape[0]).cpu().data.numpy()
    b1_model = torch.stack([tup[1] for tup in MC_SAMPLES]).view(N_MC,Y_tst.shape[0]).cpu().data.numpy()
    
    dropout_cov = np.array([calc_cov(b0_tmp, b1_tmp) for b0_tmp, b1_tmp in zip(b0_model.T, b1_model.T)])
    dropout_cov = np.mean(dropout_cov, 0)        
    # calculate frobenius norm for the difference
    fnorm = np.sqrt(np.sum((dropout_cov - pred_1_cov).flatten()**2))
    return pred_1_cov, dropout_cov, fnorm
    


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#    model_name = 'gal_concrete1'
    model_name = 'gal_model'
    S_MAX = 0.5
    model = torch.load('{}_{:.3f}.pkl'.format(model_name, S_MAX), map_location=device)
    pred_1_cov, dropout_cov, fnorm = test_covariances(model, S_MAX)
    print('predictive covariance {} dropout covariance {} fnorm {}'.format(pred_1_cov, dropout_cov, fnorm))

