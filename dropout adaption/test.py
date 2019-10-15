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
    
def test_model(model, S_MAX):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_TST = 500
    N_MC = 50
    
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
    
#    pylab.figure()
#    pylab.scatter(b0s_tst_1, b0s_pred_1)
#    pylab.figure()
#    pylab.scatter(b1s_tst_1, b1s_pred_1)
#    pylab.show()
    
    pred_1_cov = calc_cov(b0s_pred_1-b0s_tst_1, b1s_pred_1-b1s_tst_1)
    print(pred_1_cov)
    #ps = np.logspace(np.log10(.00005), np.log10(.5), 50)
    #ps = np.linspace(.00005, .25, 50)
    ps = np.linspace(.000005, .15, 300)
    ys = np.zeros_like(ps)
    
    for i, p in enumerate(ps):
        model.set_p(p=p)
        model = model.to(device)
    
        MC_SAMPLES = [model(Variable(torch.FloatTensor(Y_tst)).to(device)) for _ in range(N_MC)]
        b0_model = torch.stack([tup[0] for tup in MC_SAMPLES]).view(N_MC,Y_tst.shape[0]).cpu().data.numpy()
        b1_model = torch.stack([tup[1] for tup in MC_SAMPLES]).view(N_MC,Y_tst.shape[0]).cpu().data.numpy()
        
        dropout_cov = np.array([calc_cov(b0_tmp, b1_tmp) for b0_tmp, b1_tmp in zip(b0_model.T, b1_model.T)])
        dropout_cov = np.mean(dropout_cov, 0)        
        # calculate frobenius norm for the difference
        ys[i] = np.sqrt(np.sum((dropout_cov - pred_1_cov).flatten()**2))
        del MC_SAMPLES, b0_model, b1_model, dropout_cov
    return ps, ys
    


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_MAX = 0.5
    model=torch.load('testmodel_{:.3f}.pkl'.format(S_MAX), map_location=device)
    model.set_p(p=1e-10)
    ps, ys_050 = test_model(model, 0.5)
#    model.set_p(p=1e-10)
#    ps, ys_05_005 = test_model(model, 0.05)
    
    S_MAX = 0.25
    model=torch.load('testmodel_{:.3f}.pkl'.format(S_MAX), map_location=device)
    model.set_p(p=1e-10)
    ps, ys_025 = test_model(model, 0.25)
    
    S_MAX = 0.05
    model=torch.load('testmodel_{:.3f}.pkl'.format(S_MAX), map_location=device)
#    model.set_p(p=1e-10)
#    ps, ys_005_05 = test_model(model, 0.5)
    model.set_p(p=1e-10)
    ps, ys_005 = test_model(model, 0.05)
    
    print('For s = {:.2f} the optimal dropout rate is: p = {:1.1e}'.format(0.05, ps[np.argmin(ys_005)]))
    print('For s = {:.2f} the optimal dropout rate is: p = {:1.1e}'.format(0.25, ps[np.argmin(ys_025)]))
    print('For s = {:.2f} the optimal dropout rate is: p = {:1.1e}'.format(0.50, ps[np.argmin(ys_050)]))
    
    pylab.figure()
    pylab.plot(ps, ys_050, '*-')
    pylab.plot(ps, ys_025, '*-')
#    pylab.plot(ps, ys_05_005, '*-')
#    pylab.plot(ps, ys_005_05, '*-')
    pylab.plot(ps, ys_005, '*-')
    pylab.legend(['s = 0.5', 's = 0.25', 's = 0.05'])
#    pylab.legend(['s_train=0.5, s_appl=0.5', 's_train=0.5, s_appl=0.05', 's_train=0.05, s_appl=0.5', 's_train=0.05, s_appl=0.05'])
    pylab.xlabel('dropout rate p')
    pylab.ylabel(r'$||cov^{s}_{T_2}-cov^{s}_{T_1}||_F$')
    pylab.show()

'''
b0_ls = np.zeros(N_TST)
b1_ls = np.zeros(N_TST)
s_hut_square = np.zeros(N_TST)
cov_bls = np.zeros((N_TST,2,2))
i=0
for (x, y, b0, b1) in zip(X_tst, Y_tst, b0s_tst, b1s_tst):
    n = x.shape[0]
    tmp = np.ones((len(x),2))
    tmp[:,1] = x
    b0_ls[i], b1_ls[i] = np.matmul(np.matmul(np.linalg.inv(np.matmul(tmp.T,tmp)),tmp.T),y)
    s_hut_square[i] = 1/(n-2) * np.sum((y-(b0_ls[i] + b1_ls[i]*x))**2)
    cov_bls[i,:,:] = s_hut_square[i] * np.linalg.inv(np.matmul(tmp.T,tmp))
    i +=1
  
plot_pred_vs_gt(b0s_tst, b0_ls, b0_model[0,:],'b0', 'b0')
plot_pred_vs_gt(b1s_tst, b1_ls, b1_model[0,:],'b1', 'b1')

#pylab.figure()
#for idx in range(5):
#    pylab.scatter(x, f(x, b0s[idx], b1s[idx]), alpha=0.2)
#    pylab.plot(x, f(x, b0s[idx], b1s[idx]))
#    pylab.plot(x, f(x, b0_ls[idx], b1_ls[idx]))
#    pylab.plot(x, f(x, b0_model[idx].cpu().data.numpy(), b1_model[idx].cpu().data.numpy()))
#pylab.pause(1)
#set_trace()
    
for idx in np.random.choice(range(N_TST), size=10):
    print('')
    print('For ind={:2}'.format(idx))
    print('Ground Truth:')
    print('b0 = {:.4f}'.format(b0s_tst[idx]))
    print('b1 = {:.4f}'.format(b1s_tst[idx]))
    print('s = {:.4f}'.format(ss_tst[idx]))
    print('Model:')
    pylab.figure(3)
    pylab.scatter(x, f(x, b0s_tst[idx], b1s_tst[idx]), alpha=0.8)
    for ind in range(b0_model.shape[0]):    
        pylab.plot(x, f(x, b0_model[ind,idx], b1_model[ind, idx]), color='r', alpha=0.2)
    pylab.plot(x, f(x, np.mean(b0_model[:,idx]), np.mean(b1_model[:, idx])), color='skyblue',lw=3)
    print('b0 = {:.4f} +- {:.4f}'.format(np.mean(b0_model[:,idx]), np.sqrt(np.var(b0_model[:,idx]))))
    print('b1 = {:.4f} +- {:.4f}'.format(np.mean(b1_model[:,idx]), np.sqrt(np.var(b1_model[:,idx]))))
    pylab.title('model variation')
    
    print('LSQM:')
    b_test = [b0_ls[idx], b1_ls[idx]]
    cov_test = cov_bls[idx,:,:]
#    print(cov_test)
    lsqm = np.random.multivariate_normal(mean=b_test, cov=cov_test, size=50)
    pylab.figure(4)
    for ind in range(lsqm.shape[0]):    
        pylab.plot(x, f(x, lsqm[ind,0], lsqm[ind, 1]), color='r', alpha=0.2)
    pylab.scatter(x, f(x, b0s_tst[idx], b1s_tst[idx]), alpha=0.8)
    pylab.plot(x, f(x, np.mean(lsqm[:,0]), np.mean(lsqm[:, 1])), color='skyblue',lw=3)
    print('b0 : {:.4f} +- {:.4f}'.format(np.mean(lsqm[:,0]), np.sqrt(np.var(lsqm[:,0]))))
    print('b1 : {:.4f} +- {:.4f}'.format(np.mean(lsqm[:,1]), np.sqrt(np.var(lsqm[:,1]))))
    pylab.title('least-square variation')
pylab.show()
'''

