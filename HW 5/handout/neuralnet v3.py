# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 17:43:27 2018

@author: Yash Kumar
"""

import random
import numpy as np

def createonehot(num):
    a = {}
    for i in range(num):
        rn = np.zeros(num)
        rn[i] = 1
        a[str(i)] = rn
    return a

ohd = createonehot(10)

def csvtomatrix(txt, ohd):
    lbll = []
    attl = []
    lbdic = []
    with open(txt) as f:
        data = [l.strip() for l in f.readlines()]
    for i in range(len(data)):
        temp1 = data[i].split(",")
        lbdic.append(temp1[0])
        lbll.append(ohd[temp1[0]])
        q = [1]
        for i in range(128):
            q.append(int(temp1[i+1]))
        attl.append(np.array(q.copy()))            
    return lbll,attl, lbdic

lbll, attl, lbdic = csvtomatrix("smallTrain.csv", ohd)

def init_model(D,M,K, rand_v):
    if(rand_v == 1):
        model = [np.array([[random.uniform(-0.1,0.1) for i in range(M+1)] for j in range(D)]), np.array([[random.uniform(-0.1,0.1) for i in range(D+1)] for j in range(K)])]
        model[0][:,0] = 0.0*model[0][:,0]
        model[1][:,0] = 0.0*model[1][:,0]
        return model
    else:
        return [np.zeros((D,M+1)), np.zeros((K,D+1))]
    
def sigmoid_fn(a):
    chk1 = np.array([1])
    z = 1/(1+np.exp(-a))
    return np.concatenate([chk1,z])

def softmax_fn(b):
    a = sum(np.exp(b))
    return np.exp(b)/a

def indi(a,b):
    if a==b:
        return 1
    else:
        return 0

def forwardpass(model, x, y):
    
    a = model[0] @ x
    z = sigmoid_fn(a)
    b = model[1] @ z
    y_pred = softmax_fn(b)
    J = -y @ np.log(y_pred)
    return y_pred, J, b, z, a, x

def backwardpass(model, y_pred, y, z, b, lr, i, lbll, x):
    D = model[0].shape[0]
    M = model[0].shape[1]-1
    K = model[1].shape[0]
    
    del_m1 = -np.outer(lbll[i]-y_pred, z)
    
    gy = -y/y_pred
    gb = gy @ (np.diag(y_pred)-np.outer(y_pred,y_pred))
    gbeta = np.outer(gb,z)
    gz = (model[1].T @ gb)[1:]
    ga = np.dot(gz,np.dot(z,1-z))
    galpha = np.outer(ga,x)
    
    
    del_m0 = np.zeros((D,M+1))
 
    for u in range(D):
        for j in range(M+1):
            sum_rd = 0
            for i in range(10):
                del_m0[u,j] += model[1][i,u]*(y_pred[i]-y[i])*x[j]*z[u+1]*(1-z[u+1])
    del_m0[0:] = del_m0[0:]/2
    
    
    
    return galpha, gbeta, del_m1, gy, gb, gbeta, gz, ga, del_m0

model = init_model(4, 128, 10, 2)
for i in range(2):
    y_pred, J, b, z, a, x = forwardpass(model, attl[i], lbll[i])
    galpha, gbeta, del_m1, gy, gb, gbeta, gz, ga, del_m0 = backwardpass(model, y_pred, lbll[i], z, b, 0.1, i, lbll, attl[i])
    model[0] = model[0] - 0.1*del_m0.copy()
    model[1] = model[1] - 0.1*gbeta.copy()
