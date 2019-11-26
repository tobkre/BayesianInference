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
import pylab

def fit_model(model, x_train, y_train, criterion, normalization, device, train_params):
        model = model.to(device)
        _x = Variable(torch.FloatTensor(normalization(x_train))).to(device)
        _y = Variable(torch.FloatTensor(y_train)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_params.lr)
        for epoch in range(train_params.n_epochs):
            mu, logvar, reg = model(_x)
                        
#            sum_of_square=0
#            for param in model.parameters():
#                sum_of_square += torch.sum(torch.pow(param, 2))
            
            loss = criterion(_y, mu, logvar) + reg #+ 1e-4*sum_of_square
            
            if torch.isinf(loss):
                print("Stop")
            if train_params.verbose and (epoch == 0 or epoch % 50 == 0 or epoch == train_params.n_epochs-1):
                print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch+1,train_params.n_epochs,loss))
    #            print("Example: true: {:.3f} guess:{:.3f} +- {:.3f}".format(_y[0].cpu().data.numpy()[0], mu[0].cpu().data.numpy()[0], var[0].cpu().data.numpy()[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model
    
def fit_model_adversarial(model, x_train, y_train, criterion, normalization, device, train_params, p_adv=0.01, alpha=0.5):
        epsilon = 0.01 * (x_train.max()-x_train.min()) 
        model = model.to(device);
        n_train = len(x_train)
        n_adv = int(np.max([1,n_train*0.04]))
#        n_adv = int(np.max([1,n_train*0.9]))
        adv_int = np.random.choice(n_train, n_adv, replace=False)
        x_train_adv = x_train[adv_int, :]
        y_train_adv = y_train[adv_int, :]
#        print("({:.2f}|{:.2f})".format(x_train_adv.squeeze(), y_train_adv.squeeze()))
        _x = Variable(torch.FloatTensor(normalization(x_train))).to(device)
        _y = Variable(torch.FloatTensor(y_train)).to(device)
        _y_adv = Variable(torch.FloatTensor(y_train_adv)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=train_params.lr, weight_decay=1e-4)
#        pylab.figure()
        for epoch in range(train_params.n_epochs):
            mu, logvar, reg = model(_x)
            
            # generate adversarial example
            x_adv = perturb(model, x_train_adv, y_train_adv, criterion, epsilon)
            
            _x_adv = Variable(torch.FloatTensor(normalization(x_adv))).to(device)
            
            mu_ad, logvar_ad, reg_ad = model(_x_adv)
#            pylab.scatter(epoch, np.abs(x_adv-x_train_adv))
#            sum_of_square=0
#            for param in model.parameters():
#                sum_of_square += torch.sum(torch.pow(param, 2))
#            if epoch%100==0:
#                print("mu    = {:.2f}, logvar    = {:.2f}".format(mu[adv_int].detach().numpy().squeeze(), logvar[adv_int].detach().numpy().squeeze()))
#                print("mu_ad = {:.2f}, logvar_ad = {:.2f}".format(mu_ad.detach().numpy().squeeze(), logvar_ad.detach().numpy().squeeze()))
            loss = alpha * criterion(_y, mu, logvar) + (1-alpha) * criterion(_y_adv, mu_ad, logvar_ad) + reg + reg_ad
            
            if train_params.verbose and (epoch == 0 or epoch % 400 == 0 or epoch == train_params.n_epochs-1):
                print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch+1,train_params.n_epochs,loss))
#                print("Example: true: {:.3f} guess:{:.3f} +- {:.3f}".format(_y[0].cpu().data.numpy()[0], mu[0].cpu().data.numpy()[0], np.sqrt(var[0].cpu().data.numpy())[0]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#        pylab.pause(5)
        return model
    
def perturb(model, x_train, y_train, criterion, epsilon):
        model_cp = copy.deepcopy(model)
        x = np.copy(x_train)
        x_var = Variable(torch.FloatTensor(x), requires_grad=True)
        y_var = Variable(torch.FloatTensor(y_train))
    
        scores, sigmas,_ = model_cp(x_var)
        loss = criterion(y_var, scores, sigmas)
        loss.backward()
        grad_sign = x_var.grad.data.cpu().sign().numpy()
        x_adv = x + epsilon*grad_sign
        return x_adv