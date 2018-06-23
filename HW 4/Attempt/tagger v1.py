# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:10:31 2018

@author: Yash Kumar
"""
import numpy as np
import math




def tsvtomatrix(txt):
    with open(txt) as f:
        data = [l.strip() for l in f.readlines()]
        
    a = []
    for i in range(len(data)):
        temp1 = data[i].split("\t")
        a.append(temp1)
    return a



def matrixtoform(a):
    x = len(a)
    lbid = {}
    atid = {}
    dix = {}
    dout = {}
    c = 0
    b = 0
    lbll = []
    for i in range(x):
        if i == 0 and a[0][0]!="":
            atid[a[i][0]] = 0
            c = c+1
            lbid[a[i][1]] = 0
            b = b+1
            dix[a[i][0]] = [i]
            dout[a[i][0]] = [a[i][1]]
            lbll.append(a[i][1])
        elif i!=0 and a[i][0]!="":
            if a[i][0] not in atid:
                atid[a[i][0]] = c
                c = c+1
                dix[a[i][0]] = [int(i)]
                dout[a[i][0]] = [str(a[i][1])]
            else:
                dix[a[i][0]].append(i)
                dout[a[i][0]].append(a[i][1])
            if a[i][1] not in lbid:
                lbid[a[i][1]] = b
                b = b+1
                lbll.append(a[i][1])
    return lbll,lbid,atid,dix,dout

yo = tsvtomatrix("train.tsv")

lbll,lbid,atid,dix,dout = matrixtoform(yo)



def den_pyikxi(theta, at_lb, atid, lbid):
    m = len(atid)
    den = 0
    for i in range(len(theta)):
        j = atid[at_lb[0]]
        den1 = np.exp(theta[i][j]+theta[i][m])
        den+=den1
    return den




def num_pyikxi(theta, k, i, m):
    num = np.exp(theta[k][i]+theta[k][m])
    return num



def iyik(a,b):
    if a==b:
        return 1
    else:
        return 0
            

def start_train1(matlist, lbid, atid, lr):
    m = len(atid)
    n = len(lbid)
    theta = np.zeros((n,m+1))
    for counter in range(len(matlist)):
#        print(counter)
        if matlist[counter][0] != "":            
            i = atid[matlist[counter][0]]
            den = den_pyikxi(theta, matlist[counter], atid, lbid)
            yi = lbid[matlist[counter][1]]
            for k in range(n):
            
#                k = lbid[matlist[counter][1]]
    #            print(num_pyikxi(theta, matlist[counter], atid, lbid))
                nlr = iyik(yi, k)-num_pyikxi(theta, k, i, m)/den
                theta[k,i] = theta[k,i] + lr*nlr
                theta[k,m] = theta[k,m] + lr*nlr
    return theta
            

a = start_train1(yo, lbid, atid, 0.5)






                

#
#
#print(np.zeros((len(lbid),len(atid+1))))
def matrixtoform2(a):
    x = len(a)
    lbid = {}
    atid = {}
    b = 0
    atid["BOS"]=0
    atid["EOS"]=1
    c = 2
    for i in range(x):
        if i == 0 and a[0][0]!="":
            atid[a[i][0]] = 2
            c = c+1
            lbid[a[i][1]] = 0
            b = b+1
        elif i!=0 and a[i][0]!="":
            if a[i][0] not in atid:
                atid[a[i][0]] = c
                c = c+1
            if a[i][1] not in lbid:
                lbid[a[i][1]] = b
                b = b+1
    return lbid,atid
        




def den_pyikxi2(theta, i0, i1, i2, atid, lbid):
    m = len(atid)
    den = 0
    for j in range(len(theta)):
        den1 = np.exp(theta[j][i0]+theta[j][i1]+theta[j][i2]+theta[j][3*m])
        den+=den1
    return den

def num_pyikxi2(theta, k, i0, i1, i2, m):
    num = np.exp(theta[k][i0]+theta[k][i1]+theta[k][i2]+theta[k][3*m])
    return num



def start_train2(matlist, lbid, atid, lr):
    m = len(atid)
    n = len(lbid)
    theta = np.zeros((n,3*m+1))
    tottrn = len(matlist)
    for counter in range(tottrn):
        print(counter)
        if matlist[counter][0] != "":
            w1 = matlist[counter][0]
            if counter == 0:
                w0 = "BOS"
                if matlist[counter+1][0] != "":
                    w2 = matlist[counter+1][0]
                else:
                    w2 = "EOS"
                
            elif counter != 0 and counter != tottrn-1:
                if matlist[counter-1][0]=="":
                    w0 = "BOS"
                else:
                    w0 = matlist[counter-1][0]
                if matlist[counter+1][0]=="":
                    w2 = "EOS"
                else:
                    w2 = matlist[counter+1][0]
            
            else:
                w2 = "EOS"
                if matlist[counter-1][0]=="":
                    w0 = "BOS"
                else:
                    w0 = matlist[counter-1][0]
            i0 = 3*atid[w0]
            i1 = 3*atid[w1]+1
            i2 = 3*atid[w2]+2   
            den = den_pyikxi2(theta, i0, i1, i2, atid, lbid)
            yi = lbid[matlist[counter][1]]
            for k in range(n):
                nlr = iyik(yi, k)-num_pyikxi2(theta, k, i0, i1, i2, m)/den
                theta[k,i0] = theta[k,i0] + lr*nlr
                theta[k,i1] = theta[k,i1] + lr*nlr
                theta[k,i2] = theta[k,i2] + lr*nlr
                theta[k,3*m] = theta[k,3*m] + lr*nlr
    return theta

lbid2, atid2 = matrixtoform2(yo)

a2 = start_train2(yo, lbid2, atid2, 0.5)


def num_log1(theta, k, i, m):
    return theta[k][i]+theta[k][m]

def num_log2(theta, k, i0, i1, i2, m):
    return theta[k][i0]+theta[k][i1]+theta[k][i2]+theta[k][3*m]

def negloglik1(matlist, theta, atid, lbid):
    N = len(matlist)
    K = len(theta)
    m = len(atid)
    countn = N
    suml = 0
    for counter in range(N):
        if matlist[counter][0]!="":
            den = den_pyikxi(theta, matlist[counter], atid, lbid)
            i = atid[matlist[counter][0]]
            for k in range(K):
                yi = lbid[matlist[counter][1]]
                suml += iyik(k, yi)*(num_log1(theta, k, i , m)-np.log(den))
        else:
            countn-=1
    return -suml/countn

print(negloglik1(yo, a, atid, lbid))


def negloglik2(matlist, theta, atid2, lbid2):
    N = len(matlist)
    K = len(theta)
    m = len(atid2)
    countn = N
    suml = 0
    for counter in range(N):
        if matlist[counter][0] != "":
            w1 = matlist[counter][0]
            if counter == 0:
                w0 = "BOS"
                if matlist[counter+1][0] != "":
                    w2 = matlist[counter+1][0]
                else:
                    w2 = "EOS"
            elif counter != 0 and counter != N-1:
                if matlist[counter-1][0]=="":
                    w0 = "BOS"
                else:
                    w0 = matlist[counter-1][0]
                if matlist[counter+1][0]=="":
                    w2 = "EOS"
                else:
                    w2 = matlist[counter+1][0]
            else:
                w2 = "EOS"
                if matlist[counter-1][0]=="":
                    w0 = "BOS"
                else:
                    w0 = matlist[counter-1][0]
            i0 = 3*atid2[w0]
            i1 = 3*atid2[w1]+1
            i2 = 3*atid2[w2]+2
            den = den_pyikxi2(theta, i0, i1, i2, atid2, lbid2)
            for k in range(K):
                yi = lbid2[matlist[counter][1]]
                suml += iyik(k, yi)*(num_log2(theta,k,i0,i1,i2,m)-np.log(den))
        else:
            countn -= 1
    return -suml/countn

print(negloglik2(yo, a2, atid2, lbid2))       