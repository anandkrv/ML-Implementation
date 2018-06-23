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

import argparse

import random
import numpy as np

#parser = argparse.ArgumentParser()
#
#parser.add_argument("traininput")
#parser.add_argument("validinput")
#parser.add_argument("trainout")
#parser.add_argument("metricsout")
#parser.add_argument("numepoch")
#parser.add_argument("hiddenu")
#parser.add_argument("initflag")
#parser.add_argument("lr")
#
#args = parser.parse_args()

def createonehot(num):
    a = {}
    for i in range(num):
        rn = np.zeros(num)
        rn[i] = 1
        a[str(i)] = rn
    return a



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

    gb = -(y-y_pred)
    gbeta = np.outer(gb,z)
    gz = (model[1][:,1:].T @ gb)
    ga = gz*z[1:]*(1-z[1:])
    galpha = np.outer(ga,x)
    return galpha, gbeta, del_m1, gb, gz, ga, del_m0
        
def predict(model, x, y):
    a = model[0] @ x
    z = sigmoid_fn(a)
    b = model[1] @ z
    y_pred = softmax_fn(b)
    return np.argmax(y_pred)


ohd = createonehot(10)

lbll, attl, lbdic = csvtomatrix("smallTrain.csv", ohd)
lbll2, attl2, lbdic2 = csvtomatrix("smallValidation.csv",ohd)
model = init_model(4, 128, 10, 2)


cet = []
cev = []
for ac in range(2):
    suma = 0
    sumv = 0
    for i in range(500):    
        y_pred, J, b, z, a, x = forwardpass(model, attl[i], lbll[i])
        galpha, gbeta, del_m1, gb, gz, ga, del_m0 = backwardpass(model, y_pred, lbll[i], z, b, 0.1, i, lbll, attl[i])
        qqq=1
        model[0] = model[0] - 0.1*galpha.copy()
        model[1] = model[1] - 0.1*gbeta.copy()
    
    for i in range(500):
        _,J,_,_,_,_ =     forwardpass(model, attl[i], lbll[i])
        suma += J
        
    for i in range(len(lbll2)):
        _,J,_,_,_,_ =     forwardpass(model, attl2[i], lbll2[i])
        sumv += J
        
    cet.append(suma/500)
    cev.append(sumv/100)

errtrn = 0    
with open("trainout.csv", "w+") as f:
    for i in range(len(lbll)):
        a = predict(model, attl[i], lbll[i])
        f.write(str(a))
        f.write("\n")
        if a != int(lbdic[i]):
            errtrn += 1
errtrn = errtrn/len(lbll)

errvld = 0        
with open("validout.csv", "w+") as f:
    for i in range(len(lbll2)):
        a = predict(model, attl2[i], lbll2[i])
        f.write(str(a))
        f.write("\n")
        if a != int(lbdic2[i]):
            errvld += 1
errvld = errvld/len(lbll2)

with open("metricsout.csv","w+") as f:
    for i in range(len(cet)):
        f.write("epoch="+str(i+1)+" crossentropy(train): "+str(cet[i]))
        f.write("\n")
        f.write("epoch="+str(i+1)+" crossentropy(validation): "+str(cev[i]))
        f.write("\n")
    f.write("error(train): "+str(errtrn)+"\n")
    f.write("error(validation): "+str(errvld)+"\n")

#with open("metricsout.csv", "w+") as f:
    