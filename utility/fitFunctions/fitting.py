# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:39:28 2019

@author: kretz01
"""
import copy
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np

def fit_model(model, x_train, y_train, n_epoch, criterion, learning_rate, normalization, device, verbose):
        model = model.to(device)
        _x = Variable(torch.FloatTensor(normalization(x_train))).to(device)
        _y = Variable(torch.FloatTensor(y_train)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        for epoch in range(n_epoch):
            mu, sig = model(_x)
                        
#            sum_of_square=0
#            for param in model.parameters():
#                sum_of_square += torch.sum(torch.pow(param, 2))
            
            loss = criterion(_y, mu, sig) #+ 1e-4*sum_of_square
            
            if torch.isinf(loss):
                print("Stop")
            if verbose and (epoch == 0 or epoch % 50 == 0 or epoch == n_epoch-1):
                print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch+1,n_epoch,loss))
    #            print("Example: true: {:.3f} guess:{:.3f} +- {:.3f}".format(_y[0].cpu().data.numpy()[0], mu[0].cpu().data.numpy()[0], var[0].cpu().data.numpy()[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model
    
def fit_model_adversarial(model, x_train, y_train, n_epoch, criterion, learning_rate, normalization, device, verbose, alpha=0.5):
        epsilon = 0.01 * (x_train.max()-x_train.min()) 
        model = model.to(device);
        x_adv = perturb(model, x_train, y_train, criterion, epsilon)
        _x = Variable(torch.FloatTensor(normalization(x_train))).to(device)
        _x_adv = Variable(torch.FloatTensor(normalization(x_adv))).to(device)
        _y = Variable(torch.FloatTensor(y_train)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        for epoch in range(n_epoch):
            mu, var = model(_x)
            mu_ad, var_ad = model(_x_adv)
            
#            sum_of_square=0
#            for param in model.parameters():
#                sum_of_square += torch.sum(torch.pow(param, 2))
            
            loss = alpha * criterion(_y, mu, var) + (1-alpha) * criterion(_y, mu_ad, var_ad) #+ 1e-4*sum_of_square
            
            if verbose and (epoch == 0 or epoch % 50 == 0 or epoch == n_epoch-1):
                print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch+1,n_epoch,loss))
#                print("Example: true: {:.3f} guess:{:.3f} +- {:.3f}".format(_y[0].cpu().data.numpy()[0], mu[0].cpu().data.numpy()[0], np.sqrt(var[0].cpu().data.numpy())[0]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model
    
def perturb(model, x_train, y_train, criterion, epsilon):
        model_cp = copy.deepcopy(model)
        x = np.copy(x_train)
        x_var = Variable(torch.FloatTensor(x), requires_grad=True)
        y_var = Variable(torch.FloatTensor(y_train))
    
        scores, sigmas = model_cp(x_var)
        loss = criterion(y_var, scores, sigmas)
        loss.backward()
        grad_sign = x_var.grad.data.cpu().sign().numpy()
        x_adv = x + epsilon*grad_sign
        return x_adv