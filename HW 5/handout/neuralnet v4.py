# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 23:17:13 2018

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
#    lbdic = []
    with open(txt) as f:
        data = [l.strip() for l in f.readlines()]
    for i in range(len(data)):
        temp1 = data[i].split(",")
#        lbdic.append(temp1[0])
        lbll.append(ohd[temp1[0]])
        q = []
        for i in range(128):
            q.append(int(temp1[i+1]))
        q.append(1)
        attl.append(np.array(q.copy()))            
    return lbll,attl

def init_model(D,M,K, rand_v):
    if(rand_v == 1):
        model = [np.array([[random.uniform(-0.1,0.1) for i in range(M+1)] for j in range(D)]), np.array([[random.uniform(-0.1,0.1) for i in range(D+1)] for j in range(K)])]
        model[0][:,0] = 0.0*model[0][:,0]
        model[1][:,0] = 0.0*model[1][:,0]
        return model
    else:
        return np.zeros((D,M+1)), np.zeros((K,D+1))

def sigmoid_fn(a):
    chk1 = np.array([1])
    z = 1/(1+np.exp(-a))
    return np.concatenate([z,chk1])

def softmax_fn(b):
    a = sum(np.exp(b))
    return np.exp(b)/a

def forwardpass(alpha, beta, x, y):
    a = alpha @ x
    z = sigmoid_fn(a)
    b = beta @ z
    y_pred = softmax_fn(b)
    J = -y @ np.log(y_pred)
    return y_pred.copy(), J.copy(), b.copy(), z.copy(), a.copy()

def backwardpass(alpha, beta, y_pred, y, z, b, lr, x):
    D = alpha.shape[0]
    M = alpha.shape[1]-1
    K = beta.shape[0]
#    print(D,M,K)
    del_m1 = np.outer(y_pred-y, z)
#    print(del_m1)
    del_m0 = np.zeros((D,M+1))
#    print(del_m0)
    for u in range(D):
        for j in range(M+1):
            sum_rd = 0
            for k in range(K):
                sum_rd += beta[k,u]*(y_pred[k]-y[k])
            del_m0[u,j] += sum_rd.copy()*x[j]*z[u]*(1-z[u])
#    print(del_m0)
    return del_m1.copy(), del_m0.copy()

lbll, attl = csvtomatrix("smallTrain.csv",ohd)
alpha, beta = init_model(4, 128, 10, 2)

sum1 = 0
for i in range(500):
    y_pred, J, b, z, a = forwardpass(alpha, beta, attl[i], lbll[i])
    del_m1, del_m0 = backwardpass(alpha, beta, y_pred, lbll[i], z, b, 0.1, attl[i])
    qqq=1
    alpha = alpha.copy() - 0.1*del_m0.copy()
    beta = beta.copy() - 0.1*del_m1.copy()
    
    
for i in range(500):
    _,J,_,_,_ =     forwardpass(alpha, beta, attl[i], lbll[i])
    sum1 += J
    
print(sum1/500)




