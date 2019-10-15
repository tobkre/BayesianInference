# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:49:54 2019

@author: kretz01
"""
import numpy as np
np.random.seed(0)

class myData():
    def __init__(self, N, b0_max=5, b1_max=5, s_max=0.5, s_pol='constant', n_x=50):
        self.b0s = np.array([(-1)**np.random.randint(0,2) * b0_max*np.random.rand() for _ in range(N)])
        self.b1s = np.array([(-1)**np.random.randint(0,2) * b1_max*np.random.rand() for _ in range(N)])
        if s_pol=='constant':
            self.ss = s_max * np.ones((N,))
        else:
            self.ss = np.array([s_max*np.random.rand() for _ in range(N)])
        
        self.X = np.array([(i-1)/(n_x-1) for i in range(1,n_x+1)])
            
    def get_data(self):
        data = np.array([self.__gen_data(b0, b1, s) for (b0, b1, s) in zip(self.b0s,self.b1s,self.ss)])
        X = np.array([point[0,:] for point in data])
        Y = np.array([point[1,:] for point in data])
        return X, Y, self.b0s, self.b1s, self.ss

    def __gen_data(self, b0, b1, s):
        Y = self.f(b0, b1)
        Y = Y + s * np.random.standard_normal(size=self.X.shape)
        return self.X, Y

    def f(self, b0, b1):
        return b0 + b1 * self.X
    
if __name__=='__main__':
    N = 100
    data = myData(N=N)
    X, Y, b0s, b1s, ss = data.get_data()