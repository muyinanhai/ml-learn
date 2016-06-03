#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################
##Author: Vincent.Y
################################

import numpy as np

def sigmoid(X):
    return 1.0/(1.0+np.exp(-X))

class LogisticRegression(object):
    """
    solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’,'sgd'}
        sgd support only

    alpha:float
        L1
    lam: float
        L2
    max_iter:int
        iteration of for the solvers to converge
    """
    def __init__(self,solver="sgd",alpha=0,lam=1,lr=0.2,max_iter=200,bias=False):
        self.solver=solver
        self.coef_=None
        self.bias=bias
        self.lam=lam
        self.alpha=alpha
        if self.solver=='sgd':
            self.lr=lr
            self.max_iter=max_iter

    def gradient_descent(self,X,y):
        m=len(y)
        for i in xrange(0,self.max_iter):
            pred=sigmoid(X.dot(self.coef_))
            for j in xrange(0,X.shape[1]):
                tmp=X[:,j]
                errors = np.mean((pred - y) * tmp) + 2*self.lam*(self.coef_[j] if j< X.shape[1] else 0) + self.alpha*(0 if self.coef_[j]==0 else 1)
                self.coef_[j]=self.coef_[j] - self.lr * errors
        return self.coef_

    def fit(self,X,y):
        if self.bias:
            X = np.hstack([X,np.ones((X.shape[0],1))])

        if self.solver=="ls":
            G = self.lam * np.eye(X.shape[1])
            G[-1, -1] = 0  # Don't regularize bias
            self.coef_=np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)),np.dot(X.T, y))
        else:
            self.coef_=np.zeros(X.shape[1])
            self.coef_=self.gradient_descent(X,y)

    def predict_proba(self,X):
        if self.bias:
            X = np.hstack([X,np.ones((X.shape[0],1))])
        return sigmoid(X.dot(self.coef_))

    def predict(self,X):
        if self.bias:
            X = np.hstack([X,np.ones((X.shape[0],1))])
        return np.array([ 1 if i>0.5 else 0 for i in sigmoid(X.dot(self.coef_))])


if __name__=="__main__":
    x=np.array([1,2,3])
    x=x.reshape(-1,1)
    y=np.array([0,0,1])

    model=LogisticRegression(bias=True,solver='sgd',max_iter=100,lam=0)
    model.fit(x,y)
    print model.coef_
    print model.predict(x)
