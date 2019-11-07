# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:11:09 2019

@author: kretz01
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import pylab

class myNet(nn.Module):
    def __init__(self, nb_features):
        super(myNet, self).__init__()
        
        self.linear1 = nn.Linear(1, nb_features, bias=True)
#        self.linear2 = nn.Linear(nb_features, nb_features)
        self.linear3 = nn.Linear(nb_features, 1, bias=False)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.linear1(x))
#        x2 = self.relu(self.linear2(x1))
        xout = self.linear3(x1)    
        return xout

def mse_loss(y, y_hat):
    return torch.pow(y-y_hat,2)

N = 40
N_EPOCHS = 750
LR = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon = np.random.normal(loc=0, scale=1, size=(N,1))*3
x_train = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
y_train = x_train**3 + epsilon
criterion = mse_loss
model = myNet(100).to(device)
_x = Variable(torch.FloatTensor(x_train)).to(device)
_y = Variable(torch.FloatTensor(y_train)).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
for epoch in range(N_EPOCHS):
    mu = model(_x)
    loss = criterion(_y, mu)
    avg_loss = torch.mean(loss)
#    print(avg_loss.cpu().data.numpy())
    optimizer.zero_grad()
    avg_loss.backward(retain_graph=True)
    optimizer.step()
    
#model.eval()

y_pred = model(_x)
grads = torch.zeros((40,300))

pylab.figure()
pylab.scatter(x_train, y_train, color='r')
pylab.plot(x_train, y_pred.cpu().data.numpy(), color='b')

for i in range(y_pred.shape[0]):
    model.zero_grad()
    y_pred[i].backward(retain_graph=True)

    grad_y = torch.empty(0)
    for params in model.parameters():
        if i==0:
            print(params.shape)
        grad_y = torch.cat((grad_y, params.grad.flatten()))
    grads[i,:] = grad_y

A = np.matmul(grads.cpu().data.numpy().T,grads.cpu().data.numpy())+1e-16
B = np.linalg.inv(np.diag(np.diag(A)))
pylab.matshow(np.matmul(B,A))
pylab.colorbar()

pylab.matshow(np.log(np.abs(A)+1e-12))
pylab.colorbar()
