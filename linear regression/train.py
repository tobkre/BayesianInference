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

from model_onep import myNet

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 10000
N_TST = 500
N_MC = 100
N_EPOCHS = 10000
N_PRINT = 100

s = 0.5

#model_name = 'gal_concrete1609'
model_name = 'gal_fixed0810'

b0s = np.array([(-1)**np.random.randint(0,2) * 5*np.random.rand() for _ in range(N)])
b1s = np.array([(-1)**np.random.randint(0,2) * 5*np.random.rand() for _ in range(N)]) 
#ss = np.array([0.5*np.random.rand() for _ in range(N)])
ss = s*np.ones((N,))

b0s_tst = np.array([(-1)**np.random.randint(0,2) * 5*np.random.rand() for _ in range(N_TST)]) 
b1s_tst = np.array([(-1)**np.random.randint(0,2) * 5*np.random.rand() for _ in range(N_TST)]) 
#ss_tst = np.array([0.5*np.random.rand() for _ in range(N_TST)])
ss_tst = s*np.ones((N_TST,))

def plot_pred_vs_gt(gt, pred_lsqm, pred_model, title=''):
    _, (ax1, ax2) = pylab.subplots(1,2, sharex=True, sharey=True)
    ax1.scatter(gt, pred_lsqm, alpha=0.2)
    ax1.set_title(r'${}$ lsqm'.format(title))
    ax1.set_xlabel('ground truth')
    ax1.set_ylabel('prediction')
    ax2.scatter(gt, pred_model, alpha=0.2)
    ax2.set_title(r'${}$ neural net'.format(title))
    ax2.set_xlabel('ground truth')
    ax2.set_ylabel('prediction')

def f(X, b0, b1):
    return b0 + b1 * X

def gen_data(b0, b1, s, n=50):
    X = np.array([(i-1)/(n-1) for i in range(1,n+1)])
    Y = f(X, b0, b1)
    Y = Y + s * np.random.standard_normal(size=X.shape)
    return X, Y

def myloss(true, x, b0, b1):
    guess = torch.mul(x, b1) + b0
    return torch.mean(torch.pow(true-guess,2))

def fit_model(n_epoch, X, Y):
    _x = Variable(torch.FloatTensor(X)).to(device)
    _y = Variable(torch.FloatTensor(Y)).to(device)
    model = myNet(1024).to(device);
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        b0_model, b1_model, reg = model(_y)
        
        loss = myloss(_y, _x, b0_model, b1_model) + reg 
#        loss = myloss(_y, _x, b0_model, b1_model) 
        if epoch % N_PRINT == 0:
#            print(torch.sigmoid(model.p_logit))
            print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch,n_epoch,loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
    
data = np.array([gen_data(b0, b1, s) for (b0, b1, s) in zip(b0s,b1s,ss)])
X = np.array([point[0,:] for point in data])
Y = np.array([point[1,:] for point in data])

data_tst = np.array([gen_data(b0, b1, s) for (b0, b1, s) in zip(b0s_tst,b1s_tst, ss_tst)])
X_tst = np.array([point[0,:] for point in data_tst])
Y_tst = np.array([point[1,:] for point in data_tst])

# train model
model = fit_model(N_EPOCHS, X, Y)
# save model
torch.save(model, '{}_{:.3f}.pkl'.format(model_name, s))

#sample model with varying dropout
MC_SAMPLES = [model(Variable(torch.FloatTensor(Y_tst)).to(device)) for _ in range(N_MC)]
b0_model = torch.stack([tup[0] for tup in MC_SAMPLES]).view(N_MC,Y_tst.shape[0]).cpu().data.numpy()
b1_model = torch.stack([tup[1] for tup in MC_SAMPLES]).view(N_MC,Y_tst.shape[0]).cpu().data.numpy()

# least squares solution
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

# save testdata
np.savez('testdata_{}_{:.3f}.npz'.format(model_name, s), b0_gt=b0s_tst, b1_gt=b1s_tst, b0_model=b0_model, b1_model=b1_model, b0_ls=b0_ls, b1_ls=b1_ls)

print('')
print('Model parameters:')
for module in model.modules():
    if hasattr(module, 'p_logit'):
        print('drop rate = {:1.1e}'.format(torch.sigmoid(module.p_logit).cpu().data.numpy().flatten()[0]))
    
plot_pred_vs_gt(b0s_tst, b0_ls, b0_model[0,:], title='b_0')
plot_pred_vs_gt(b1s_tst, b1_ls, b1_model[0,:], title='b_1')

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
    lsqm = np.random.multivariate_normal(mean=b_test, cov=cov_test, size=N_MC)
    pylab.figure(4)
    for ind in range(lsqm.shape[0]):    
        pylab.plot(x, f(x, lsqm[ind,0], lsqm[ind, 1]), color='r', alpha=0.2)
    pylab.scatter(x, f(x, b0s_tst[idx], b1s_tst[idx]), alpha=0.8)
    pylab.plot(x, f(x, np.mean(lsqm[:,0]), np.mean(lsqm[:, 1])), color='skyblue',lw=3)
    print('b0 : {:.4f} +- {:.4f}'.format(np.mean(lsqm[:,0]), np.sqrt(np.var(lsqm[:,0]))))
    print('b1 : {:.4f} +- {:.4f}'.format(np.mean(lsqm[:,1]), np.sqrt(np.var(lsqm[:,1]))))
    pylab.title('least-square variation')
pylab.show()