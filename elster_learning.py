# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:43:14 2019

@author: kretz01
"""
import copy
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim
import pylab

from utility.lossFunctions.loss import nll_loss

class elsterLearning:
    def __init__(self, weight_regularizer=1e-3, N_samples=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_regularizer = weight_regularizer
        
        
        self.N = N_samples
        self.is_trained = False
        
    def train(self, model_class, n_features, x_train, y_train, train_params):
        x = Variable(torch.FloatTensor(x_train)).to(self.device)
        y = Variable(torch.FloatTensor(y_train)).to(self.device)
        self.model = model_class(n_features, weight_regularizer = self.weight_regularizer)
        self.p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        optimizer = optim.Adam(self.model.parameters(), lr=train_params.lr)
        for epoch in range(train_params.n_epochs):
            y_hat, logvar_hat, reg = self.model(x)
            loss = nll_loss(y, y_hat,logvar_hat) + reg
            if train_params.verbose==True and (epoch==0 or epoch%400==0 or epoch==train_params.n_epochs-1):
                print("Epoch {:d}/{:d}, Loss={:.4f}".format(epoch+1,train_params.n_epochs,loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.is_trained = True
        self.theta = np.empty(0)
        for param in self.model.parameters():
            self.theta = np.concatenate((self.theta, param.flatten().cpu().data.numpy()))
        y_pred, logvar_pred, _ = self.model(x)
        self.tau_map = self.__calc_tau_map(y_train, y_pred, logvar_pred)
        
        self.jac = np.zeros((y_pred.shape[0],self.p))   
        for i in range(y_pred.shape[0]):
            self.model.zero_grad()
            y_pred[i].backward(retain_graph=True)
            grad_y = torch.empty(0)
            for params in self.model.parameters():
                grad_y = torch.cat((grad_y, params.grad.flatten()))
            self.jac[i,:] = grad_y.cpu().data.numpy()
        self.delta_theta = self.sample_delta_theta(x.cpu().data.numpy(), y.cpu().data.numpy())
        
    def __calc_tau_map(self, y_train, y_pred, logvar):
        if self.is_trained:
            y_train = torch.FloatTensor(y_train)
#            y_pred = torch.FloatTensor(y_pred)
            n = len(y_train)
            return ((1/(self.p + n - 2))*nll_loss(y_train, y_pred, logvar))**(-1)
        else:
            return 0
        
    def sample_delta_theta(self, x, y):
        reg=0
        n = len(y)
        for param in self.model.parameters():
            reg += torch.sum(torch.pow(param, 2))
        reg = self.weight_regularizer*reg.cpu().data.numpy()
        y_p= self.model(torch.FloatTensor(x))[0].cpu().data.numpy()
        gamma = (np.matmul(np.transpose(y-y_p), y-y_p)+reg).flatten()[0]
        alpha = (n+self.p)/2
        delta_tau = 10
        exp_tau = self.tau_map.cpu().data.numpy()
#        print(gamma)
        i=0
        while delta_tau > 1e-2 and i<100:
#            print(exp_tau)
            exp_tau_old = exp_tau
            var_delta = (exp_tau)**-1 * (np.diag(np.matmul(self.jac.T, self.jac))+self.weight_regularizer)**-1
            beta = 0.5 * (gamma+np.sum(var_delta*(np.diag(np.matmul(self.jac.T, self.jac))+self.weight_regularizer)))
            exp_tau = alpha/beta
            delta_tau = np.abs(exp_tau-exp_tau_old)
#            print('delta tau: {:.4f}'.format(delta_tau))
            i+=1
        delta_theta = np.random.normal(loc=0, scale=np.sqrt(var_delta), size=(self.N, len(var_delta)))
        return delta_theta
    
    def predict(self, x):
        if self.is_trained:
            y_pred = self.model(torch.FloatTensor(x))[0].cpu().data.numpy()
            return y_pred
        
#    def sample_predict(self, x):
#        y_pred = np.zeros((self.N,len(x)))
#        
#        if self.is_trained:
#            for i in range(self.N):
#                model = copy.deepcopy(self.model)
#                delta_theta = self.delta_theta[i,:].flatten()
#                print(delta_theta)
#                for param in model.parameters():
#                    mask = np.isin(self.theta, param.flatten().cpu().data.numpy())
#                    param_delta = torch.ones_like(param.data)
#                    param.data = param.data + param_delta.new_tensor(delta_theta[mask]).view(param.data.size())
#                y_pred[i,:] = model(torch.FloatTensor(x)).cpu().data.numpy().flatten()
#            return y_pred
        
    def sample_predict(self, x, y, tau):
        y_p = np.zeros_like(x)
        u_square = np.zeros_like(x)
#        U_theta = self.tau_map.cpu().data.numpy()**(-1) * np.linalg.inv(np.matmul(self.jac.T, self.jac)+self.weight_regularizer*np.eye(self.p))
        U_theta = tau**(-1) * np.linalg.inv(np.matmul(self.jac.T, self.jac)+self.weight_regularizer*np.eye(self.p))
        if self.is_trained:
            for i,ix in enumerate(x):
                y_pred = self.model(torch.FloatTensor(ix))[0]
                self.model.zero_grad()
                y_pred.backward(retain_graph=True)
                grad_y = torch.empty(0)
                for params in self.model.parameters():
                    grad_y = torch.cat((grad_y, params.grad.flatten()))
                df = grad_y.view(-1,1).cpu().data.numpy()
                
                y_p[i] = y_pred.cpu().data.numpy()
                u_square[i] = np.sqrt(np.matmul(df.T, np.matmul(U_theta, df)))
#            l = len(y_p)
#            print((np.sum(((y-y_p)**2-self.tau_map.cpu().data.numpy()**(-1))/u_square**2)-l)**2)
        return y_p, u_square, U_theta
                    
    def sample_predict1(self, x, y, tau, N=20):
        y_p = np.zeros((N,len(x)))
#        U_theta = self.tau_map.cpu().data.numpy()**(-1) * np.linalg.inv(np.matmul(self.jac.T, self.jac)+self.weight_regularizer*np.eye(self.p))
        U_theta = tau**(-1) * np.linalg.inv(np.matmul(self.jac.T, self.jac)+self.weight_regularizer*np.eye(self.p))
        delta_thetas = np.random.multivariate_normal(np.zeros((self.p,)), U_theta, N)
        if self.is_trained:
            for i in range(N):
                delta_theta = delta_thetas[i,:]
                model = copy.deepcopy(self.model)
                
                for param in model.parameters():
                    mask = np.isin(self.theta, param.data.flatten().cpu().data.numpy())
                    param_delta = torch.ones_like(param.data)
#                    param_delta = torch.rand_like(param.data)
                    param.data = param.data + param_delta.new_tensor(delta_theta[mask]).view(param.data.size())
#                    param.data = param.data + param.data * param_delta
                y_p[i,:] = model(torch.FloatTensor(x))[0].cpu().data.numpy().flatten()
#                print('Model {:.2f} and adjusted {:.2f}'.format(self.model(torch.FloatTensor(x))[0].cpu().data.numpy().flatten()[0], y_p[i,0]))
            u_square = np.std(y_p, 0)
            y_p_m = np.mean(y_p, 0)
#            print('{:.4f} +- {:.4f}'.format(y_p_m[0], u_square[0]))
        return y_p_m, u_square, U_theta

def get_dev(y_t, y_p, y_std, tau):
    l = len(y_p)
    dev = (np.sum(((y_t.squeeze()-y_p.squeeze())**2-tau**(-1))/y_std.squeeze()**2)-l)**2
    return dev

def make_plot(x, y, x_tst, y_tst, y_tst_pred, y_tst_samples, lc, uc, textstr):
    pylab.figure()
    pylab.scatter(x, y, color='r')
    pylab.plot(x_tst, y_tst)
    pylab.plot(x_tst,y_tst_pred)
#    pylab.plot(x_tst, y_tst_samples)
    pylab.fill_between(x_tst.squeeze(), lc.squeeze(), uc.squeeze(), color='gray', alpha=0.5)
    pylab.xlim([-7,7])
    pylab.ylim([-250,250])
    pylab.text(-6.8, 150., textstr, fontsize=12)
    
if __name__=='__main__':
    from utility.models.linearModels import hidden1
    from playground.ut import training_parameters
    N = 50
    np.random.seed(1)
    
#    lbda = 1.95e-1
    lbda = 1.95e-4
    tau = 0.35e+3
    
    epsilon = np.random.normal(loc=0, scale=1, size=(N,1))*3
#    epsilon=0
    x = np.sort(np.random.uniform(low=-4., high=4., size=(N,1)), axis=0)
#    x = np.linspace(start=-4, stop=4, num=N).reshape((-1,1))
    y = (x)**3 + epsilon
    
    x_tilde = np.linspace(start=-4, stop=4, num=N).reshape((-1,1))
    y_tilde = x_tilde**3 + epsilon
        
    x_tst = np.linspace(start=-6, stop=6, num=100).reshape((-1,1))
    y_tst = (x_tst)**3
    
    train_params = training_parameters(lr=0.01, n_epochs=2000, verbose=True)
    model = hidden1(100)
    
    tmp = elsterLearning(weight_regularizer=lbda)
    tmp.train(hidden1, 100, x, y, train_params)
    
    y_tst_pred = tmp.predict(x_tst)
    
    y_tilde_hat, y_tilde_std, _ = tmp.sample_predict1(x_tilde, y_tilde, tau=tau)
    eps_tilde = get_dev(y_tilde, y_tilde_hat, y_tilde_std, tau)
    print('Lambda={:.2e}, tau={:.2e}'.format(lbda, tau))
    print('epsilon = {:.2e}'.format(eps_tilde))
    
    y_tst_samples, y_tst_std, U_theta = tmp.sample_predict(x_tst, y_tst, tau=tau)
    tst = get_dev(y_tst, y_tst_samples, y_tst_std, tau)
    
    textstr = '\n'.join((
            r'$\left[\sum_{i=1}^L \frac{(\tilde{y_i} - \hat{y_i})^2-\hat{\tau}}{u^2(y_i)}-L\right]^2 = %1.4e$' % (tst),
            r'$\lambda = %1.2e$' % (tmp.weight_regularizer)))
    
    lc = y_tst_samples - y_tst_std
    uc = y_tst_samples + y_tst_std
    
    y_tst_samples1, y_tst_std1, U_theta1 = tmp.sample_predict1(x_tst, y_tst, tau=tau)
    tst1 = get_dev(y_tst, y_tst_samples1, y_tst_std1, tau)
    
    textstr1 = '\n'.join((
            r'$\left[\sum_{i=1}^L \frac{(\tilde{y_i} - \hat{y_i})^2-\hat{\tau}}{u^2(y_i)}-L\right]^2 = %1.4e$' % (tst1),
            r'$\lambda = %1.2e$' % (tmp.weight_regularizer)))
    
    lc1 = y_tst_pred.squeeze() - y_tst_std1
    uc1 = y_tst_pred.squeeze() + y_tst_std1
    
    make_plot(x, y, x_tst, y_tst, y_tst_pred, y_tst_samples, lc, uc, textstr)
    make_plot(x, y, x_tst, y_tst, y_tst_pred, y_tst_samples1, lc1, uc1, textstr1)
    
    pylab.figure()
    pylab.plot(x_tst, y_tst_pred-y_tst)
    pylab.fill_between(x_tst.squeeze(), (y_tst_pred.squeeze()-y_tst.squeeze()-y_tst_std1).squeeze(), (y_tst_pred.squeeze()-y_tst.squeeze()+y_tst_std1).squeeze(), color='gray', alpha=0.5)
    print(np.all(U_theta==U_theta1))
#    pylab.figure()
#    pylab.imshow(U_theta)
#    pylab.colorbar()
#    
#    pylab.figure()
#    pylab.imshow(U_theta1)
#    pylab.colorbar()
#    print(delta_theta.shape)
    