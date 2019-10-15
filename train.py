# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:06:30 2019

@author: kretz01
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Variable

from mynet import MyNet
from mydataset import MyDataset

from pdb import set_trace
import pylab

N_DATA = 1000
N_Val = 500
INITIAL_LEARNING_RATE = 0.001
BATCH_SIZE = 200
N_EPOCHS = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gen_data(N, Q=1, D=1):
    sigma = 2e0
    X = np.random.randn(N, Q)
    w = 2.
    b = 8.
    
    Y = X.dot(w) + b + sigma * np.random.randn(N, D)
    return X, Y

def f(x):
    epsilon = torch.from_numpy(np.random.uniform(low=0.0, high=1.0, size=x.shape[0]))
    epsilon = epsilon.type(torch.FloatTensor)
    return 2*x+8+epsilon.unsqueeze(1)

X, Y = gen_data(N_DATA+N_Val)
XTrain, YTrain = X[:N_DATA], Y[:N_DATA]
XTest, YTest = X[N_DATA:], Y[N_DATA:]

XTrain = Variable(torch.FloatTensor(XTrain))
YTrain = Variable(torch.FloatTensor(YTrain))

XTest = Variable(torch.FloatTensor(XTest))
YTest = Variable(torch.FloatTensor(YTest))

train_set = MyDataset(x=XTrain, y=YTrain)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

model = MyNet(n_in=1, n_out=1).to(device)

model.eval()
YPRED = model.forward(XTest)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
scheduler = StepLR(optimizer, 20)

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x.to(device)
        y.to(device)
#        set_trace()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        epoch_loss += loss
#        plt.figure(1)
#        plt.plot(i, loss.detach().numpy(), '*')
#        plt.pause(1)
        loss.backward()
        optimizer.step()
    epoch_loss = epoch_loss/len(train_loader)
    pylab.figure(1)
    pylab.plot(epoch, epoch_loss.detach().numpy(), 'b*-')
    pylab.pause(1)
    scheduler.step()
#        set_trace()

model.eval()
YPRED2 = model.forward(XTest)

MC_samples = [model(XTest) for _ in range(20)]
YPreds = torch.stack([tup for tup in MC_samples]).view(20, XTest.shape[0]).cpu().data.numpy()

pylab.figure(2)
pylab.plot(XTest.cpu().data.numpy(), YTest.cpu().data.numpy(), '*')
pylab.plot(XTest.cpu().data.numpy(), YPRED.detach().cpu().data.numpy(), 'o')
pylab.plot(XTest.cpu().data.numpy(), YPRED2.detach().cpu().data.numpy(), 'x')
