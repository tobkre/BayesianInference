# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:50:32 2019

@author: kretz01
"""
from pdb import set_trace

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable

import numpy as np
import pylab

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

from model import Model

#Ns = [10,25,50,100,1000,10000] # Number of data points
Ns = [25,100] # Number of data points

Ns = np.array(Ns)

#nb_epochs = [2000, 1000, 500, 200, 20, 2]
nb_epochs = [1000, 200]
nb_val_size = 100

Q = 1
D = 1
K_test = 20
nb_reps = 3

l = 1e-4

def heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(precision*(true-mean)**2+log_var,1),0)

def gen_data(N, Q=1, D=1):
    sigma = 2e0
    X = np.random.randn(N, Q)
    w = 2.
    b = 8.
    
    Y = X.dot(w) + b + sigma * np.random.randn(N, D)
    return X, Y

#X, Y = gen_data(10)
#pylab.figure(figsize=(3,1.5))
#pylab.scatter(X[:,0],Y[:,0], edgecolor='b')
#pylab.show()

def fit_model(nb_epoch, X, Y):
    nb_features = 1024
    batch_size = 20
    
    N = X.shape[0]
    wr = 1**2. / N
    dr = 2. /N
    model = Model(nb_features, wr, dr)
#    model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    
    for i in range(nb_epoch):
        old_batch=0
        for batch in range(int(np.ceil(X.shape[0]/batch_size))):
            batch = (batch+1)
            _x = X[old_batch:batch_size*batch]
            _y = Y[old_batch:batch_size*batch]
            
#            x = Variable(torch.FloatTensor(_x)).cuda()
#            y = Variable(torch.FloatTensor(_y)).cuda()
            
            x = Variable(torch.FloatTensor(_x))
            y = Variable(torch.FloatTensor(_y))
            
            mean, log_var, regularization = model(x)
            
            loss = heteroscedastic_loss(y, mean, log_var) + regularization
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a-a_max), axis=0)) + a_max

def test(YTrue, K_test, means, logvar):
    k = K_test
    N = YTrue.shape[0]
    mean = means
    logvar = logvar
    test_ll = -0.5*np.exp(-logvar)*(mean-YTrue.squeeze())**2. - 0.5*logvar - 0.5*np.log(2*np.pi)
    test_ll = np.sum(np.sum(test_ll, -1), -1)
    test_ll = logsumexp(test_ll) - np.log(k)
    pppp = test_ll / N # per point predictive probability
    rmse = np.mean((np.mean(mean,0)-YTrue.squeeze())**2.)**0.5
    return pppp, rmse

def plot(XTrain, YTrain, XVal, YVal, means):
    indx = np.argsort(XVal[:,0])
    _, (ax1, ax2, ax3, ax4) = pylab.subplots(1,4, figsize=(12,1.5), sharex=True, sharey=True)
#    set_trace()
    ax1.scatter(XTrain[:,0], YTrain[:,0], c='y')
    ax1.set_title('Train set')
    ax2.plot(XVal[indx, 0], np.mean(means, 0)[indx], color='skyblue', lw=3)
    ax2.scatter(XTrain[:,0], YTrain[:,0], c='y')
    ax2.set_title('+Predictive mean')
    for mean in means:
        ax3.scatter(XVal[:,0], mean, c='b', alpha=0.2, lw=0)
    ax3.plot(XVal[indx, 0], np.mean(means, 0)[indx], color='skyblue', lw=3)
    ax3.set_title('+MC samples on validation X')
    ax4.scatter(XVal[:,0], YVal[:,0], c='r', alpha=0.2, lw=0)
    ax4.set_title('Validation set')
    pylab.show()

results = []

for N, nb_epoch in zip(Ns, nb_epochs):
    rep_results = []
    print(nb_epoch)
    for i in range(nb_reps):
        X, Y = gen_data(N + nb_val_size)
        XTrain, YTrain = X[:N], Y[:N]
        XVal, YVal = X[N:], Y[N:]
        model = fit_model(nb_epoch, XTrain, YTrain)
        model.eval()
#        MC_samples = [model(Variable(torch.FloatTensor(XVal)).cuda()) for _ in range(K_test)]
        MC_samples = [model(Variable(torch.FloatTensor(XVal))) for _ in range(K_test)]
        means = torch.stack([tup[0] for tup in MC_samples]).view(K_test, XVal.shape[0]).cpu().data.numpy()
        logvar = torch.stack([tup[1] for tup in MC_samples]).view(K_test, XVal.shape[0]).cpu().data.numpy()
        
        pppp, rmse = test(YVal, K_test, means, logvar)
        logvar = np.mean(logvar, 0)
        
        ps = np.array([torch.sigmoid(module.p_logit).cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p_logit')])
        plot(XTrain, YTrain, XVal, YVal, means)
        rep_results += [(rmse, ps)]
    test_mean = np.mean([r[0] for r in rep_results])
    test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
    ps = np.mean([r[1] for r in rep_results], 0)
    
    print(N, nb_epoch, '-', test_mean, test_std_err, ps)
    results += [rep_results]
 