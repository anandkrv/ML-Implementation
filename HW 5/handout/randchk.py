# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:46:18 2018

@author: Yash Kumar
"""
import argparse

import random
import numpy as np


def sigmoid_fn(a):
    chk1 = np.array([1])
    z = 1/(1+np.exp(-a))
    return np.concatenate([chk1,z])

def softmax_fn(b):
    a = sum(np.exp(b))
    return np.exp(b)/a

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
    return galpha, gbeta, gb, gz, ga

x = np.array([1,1,1,0,0,1,1])
y = np.array([0,1,0])

model = [np.array([[1,1,2,-3,0,1,-3],[1,3,1,2,1,0,2],[1,2,2,2,2,2,1],[1,1,0,2,1,-2,2]]), np.array([[1,1,2,-2,1],[1,1,-1,1,2],[1,3,1,-1,1]])]

y_pred, J, b, z, a, x = forwardpass(model, x, y)

galpha, gbeta, gb, gz, ga = backwardpass(model, y_pred, y, z, b, 1, 1, 0, x)

model[1] = model[1] - gbeta
model[0] = model[0] - galpha

x = np.array([1,1,1,0,0,1,1])
y = np.array([0,1,0])

y_pred, J, b, z, a, x = forwardpass(model, x, y)