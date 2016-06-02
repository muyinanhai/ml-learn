#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################
##Author: Vincent.Y
################################

import numpy as np
class LinearRegression(object):
    """
    solver: 
        ls for least square method, sgd for gridient descent
    """
    def __init__(self,solver="ls",lr=0.2,max_iter=200,bias=False):
        self.solver=solver
        self.coef_=None
        self.bias=bias
        if self.solver=='sgd':
            self.lr=lr
            self.max_iter=max_iter

    def gradient_descent(self,X,y):
        m=len(y)
        for i in xrange(0,self.max_iter):
            pred=X.dot(self.coef_)
            for j in xrange(0,X.shape[1]):
                tmp=X[:,j]
                errors = (pred - y) * tmp#element-wise multi
                self.coef_[j]=self.coef_[j] - self.lr * np.mean(errors)
        return self.coef_

    def fit(self,X,y):
        if self.bias:
            X = np.hstack([X,np.ones((X.shape[0],1))])

        if self.solver=="ls":
            self.coef_=np.linalg.lstsq(X,y)[0]
        else:
            self.coef_=np.zeros(X.shape[1])
            self.coef_=self.gradient_descent(X,y)

    def predict(self,X):
        if self.bias:
            X = np.hstack([X,np.ones((X.shape[0],1))])

        return X.dot(self.coef_)

if __name__=="__main__":
    x=np.array([1,2,3])
    x=x.reshape(-1,1)
    y=np.array([3,5,7])

    model=LinearRegression(bias=True)
    model.fit(x,y)
    print model.coef_
    print model.predict(x)

    model=LinearRegression(bias=True,solver='sgd',max_iter=1000)
    model.fit(x,y)
    print model.coef_
    print model.predict(x)
