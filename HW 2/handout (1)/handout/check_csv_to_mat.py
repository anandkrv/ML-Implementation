# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:24:29 2018

@author: Yash Kumar
"""

def csvtomatrix(input_txt):
    f = open(input_txt)
    data = [l.strip() for l in f.readlines()]
#    print(data)
    a = []
    for i in range(len(data)):
        temp1 = data[i].split(",")
        a.append(temp1)
    
    #print(a)
    #to remove the last \n
    return a

print(csvtomatrix("small_train.csv"))

def csvtomatrix1(input_txt):
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

print(csvtomatrix1("small_train.csv"))