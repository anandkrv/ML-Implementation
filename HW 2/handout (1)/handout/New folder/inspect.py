# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:30:24 2018

@author: Yash Kumar
"""
import numpy as np
import math
import scipy
    
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("input")
parser.add_argument("output")


args = parser.parse_args()



def csvtomatrix(input_txt):
    f = open(input_txt)
    data = f.readlines()
    #print(data)
    a = []
    for i in range (len(data)):
        temp = data[i]
        a.append(temp.split(","))
    #print(a)
    for i in range(len(data)):
        if a[i][len(a[0])-1][(len(a[i][len(a[0])-1]))-1:(len(a[i][len(a[0])-1]))] == "\n":
            a[i][len(a[0])-1] = a[i][len(a[0])-1][:(len(a[i][len(a[0])-1])-1)]  #to remove the last \n
    return a

# print(csvtomatrix("education_train.csv"))
    
#data = csvtomatrix("education_train.csv")
data = csvtomatrix(args.input)
#print(len(data))
#print(data)
#
#print(data[0])
#print(len(data[0]))

def entropy_err(data):
    arb_z = data[1][len(data[0])-1]
    a = [0,0]
    for i in range(len(data)-1):
        if data[i+1][len(data[0])-1] == arb_z:
            a[0] = a[0]+1
        else:
            a[1] = a[1]+1
    if a[0]<a[1]:
        error = a[0]/float(a[0]+a[1])
    else:
        error = a[1]/float(a[0]+a[1])
    
    if a[0] == 0:
        return 0
    elif a[1] == 0:
        return 0
    else:
        return(-a[0]/float(a[0]+a[1])*np.log(a[0]/float(a[0]+a[1]))/np.log(2)-a[1]/float(a[0]+a[1])*np.log(a[1]/float(a[0]+a[1]))/np.log(2),error)
    

#print(entropy_err(data))
#arb_z_c = data[2][0]
#arb_z_d = data[2][len(data[0])-1]
#
#print (arb_z_c,arb_z_d)
#
#
#def classify(data, j):
#    arb_z_c = data[1][j-1]
#    arb_z_d = data[1][len(data[0])-1]     #creating bins of [00,01,10,11]
#    databin = [0,0,0,0]
#    for i in range(len(data)-1):
#        if data[i+1][j-1] == arb_z_c and data[i+1][len(data[0])-1] == arb_z_d:
#            databin[0] = databin[0]+1
#        elif (data[i+1][j-1] == arb_z_c) and (data[i+1][len(data[0])-1] != arb_z_d):
#            databin[1] = databin[1]+1
#        elif (data[i+1][j-1] != arb_z_c) and (data[i+1][len(data[0])-1] == arb_z_d):
#            databin[2] = databin[2]+1
#        elif  (data[i+1][j-1] != arb_z_c) and (data[i+1][len(data[0])-1] != arb_z_d):
#            databin[3] = databin[3]+1
#    return databin
#
#print(classify(data,1))

entropy, error = entropy_err(data)     
txt = ["entropy: ",str(entropy),"\n","error: ",str(error)]
def outputtxt(out_txt):
    with open(out_txt, 'w') as f:
        f.writelines(txt)

outputtxt(args.output)