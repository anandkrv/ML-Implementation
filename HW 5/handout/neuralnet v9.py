# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 18:26:02 2018

@author: Yash Kumar
"""
#
#train in
#valid in
#train out
#valid out
#metrics out
#num epoch
#hidden units
#init_flag
#learning rate

import random
import numpy as np


def create_data():
    
    attl=np.array([[1, 0,1,0,0,0], [1, 0,0,1,0,0], [1, 1,0,0,0,0], [1, 0,0,0,1,0], [1, 0,0,0,0,1]])
    lbll=np.array([[0,1], [1,0], [0,1], [1,0]])
    
    return attl, lbll

def createonehot(num):
    a = {}
    for i in range(num):
        rn = np.zeros(num)
        rn[i] = 1
        a[str(i)] = rn
    return a

ohd = createonehot(10)
#print(ohd)        

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

#lbll, attl, lbdic = csvtomatrix("smallTrain.csv", ohd)

def init_model(D,M,K, rand_v):
    if(rand_v == 1):
        model = [np.array([[random.uniform(-0.1,0.1) for i in range(M+1)] for j in range(D)]), np.array([[random.uniform(-0.1,0.1) for i in range(D+1)] for j in range(K)])]
        model[0][:,0] = 0.0*model[0][:,0]
        model[1][:,0] = 0.0*model[1][:,0]
        return model
    else:
        return [np.zeros((D,M+1)), np.zeros((K,D+1))]

#print(init_model(5,4,2,1))
#print(init_model(5,4,2,2))

#A = np.array([[1,2],[3,4]])
#print(A @ A)
        
def sigmoid_fn(a):
    chk1 = np.array([1])
    z = 1/(1+np.exp(-a))
    return np.concatenate([chk1,z])

def softmax_fn(b):
    a = sum(np.exp(b))
    return np.exp(b)/a

#a = np.array([1,2,3])
#        
#print(sum(softmax_fn(a)))

def indi(a,b):
    if a==b:
        return 1
    else:
        return 0

def forwardpass(model, x, y):
    print(attl[i].shape)
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
    
#    del_m1 = np.zeros((K,D+1))
    del_m1 = -np.outer(lbll[i]-y_pred, z)
#    for u in range(K):
#        for j in range(D+1):
#            del_m1[u,j] = lbll[i][u]*z[j]*(indi(lbdic[i],u)-y_pred[u])
#    
#    del_m0 = np.zeros((D,M+1))
#    for u in range(D):
#        for m in range(M+1):
#            del_m2[u,j] = 


    #gy = -y/y_pred
    #gb = gy @ (np.diag(y_pred)-np.outer(y_pred,y_pred))
    gb = -(y-y_pred)
    gbeta = np.outer(gb,z)
    gz = (model[1][:,1:].T @ gb)
    ga = gz*z[1:]*(1-z[1:])
    galpha = np.outer(ga,x)
#    print(model[0])
#    print(model[1])
    
    del_m0 = np.zeros((D,M+1))
 

            
    
    
    
    return galpha, gbeta, del_m1, gy, gb, gbeta, gz, ga, del_m0
        
#model = init_model(4, 128, 10, 2)
#
#y_pred, J, b, z, a, x = forwardpass(model, attl[0], lbll[0])
#galpha, gbeta, del_m1, gy, gb, gbeta, gz, ga = backwardpass(model, y_pred, lbll[0], z, b, 0.1, 0, lbll)
attl, lbll = create_data()
model = init_model(2, 5, 2, 2)

    
for i in range(4):
    y_pred, J, b, z, a, x = forwardpass(model, attl[i], lbll[i])
    galpha, gbeta, del_m1, gy, gb, gbeta, gz, ga, del_m0 = backwardpass(model, y_pred, lbll[i], z, b, 0.1, i, lbll, attl[i])
    qqq=1
    model[0] = model[0] - 0.1*galpha.copy()
    model[1] = model[1] - 0.1*gbeta.copy()


    #print(model[1])
    
#a = np.array([1,2,3,4,5,6,7,8])
#print(a[1:])
    
a = np.array([[1,2,2,3,4],[1,5,6,9,8]])
#print(a[0:,:]/2)