# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:03:40 2018

@author: Yash Kumar
"""

import numpy as np

#train input
#test input
#max depth
#train out
#test out
#metrics out


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

class node(object):
    def __init__(self):
        self.att_name = ""
        self.l_br_v = None
        self.r_br_v = None
        self.l_br = None
        self.r_br = None
        self.val = None
        
    def set_br(self,l_br,r_br):
        self.l_br = l_br
        self.r_br = r_br
        


def entropy(data):
    arb_z = data[1][len(data[0])-1]
    a = [0,0]
    for i in range(len(data)-1):
        if data[i+1][len(data[0])-1] == arb_z:
            a[0] = a[0]+1
        else:
            a[1] = a[1]+1
    if a[0] == 0:
        return 0
    elif a[1] == 0:
        return 0
    else:
        return(-a[0]/float(a[0]+a[1])*np.log(a[0]/float(a[0]+a[1]))/np.log(2)-a[1]/float(a[0]+a[1])*np.log(a[1]/float(a[0]+a[1]))/np.log(2))

def xlog2x(a):
    if a == 0:
        return 0
    elif a == 1:
        return 0
    else:
        return a*np.log(a)/np.log(2)


def cond_ent(data, j):
    arb_z_c = data[1][j-1]
    arb_z_d = data[1][len(data[0])-1]     #creating bins of [00,01,10,11] where leftbit att, rightbit result
    dbin = [0,0,0,0]
    for i in range(len(data)-1):
        if data[i+1][j-1] == arb_z_c and data[i+1][len(data[0])-1] == arb_z_d:
            dbin[0] = dbin[0]+1
        elif (data[i+1][j-1] == arb_z_c) and (data[i+1][len(data[0])-1] != arb_z_d):
            dbin[1] = dbin[1]+1
        elif (data[i+1][j-1] != arb_z_c) and (data[i+1][len(data[0])-1] == arb_z_d):
            dbin[2] = dbin[2]+1
        elif  (data[i+1][j-1] != arb_z_c) and (data[i+1][len(data[0])-1] != arb_z_d):
            dbin[3] = dbin[3]+1
    cond_ent1 = -((dbin[1]+dbin[0])/float(dbin[1]+dbin[0]+dbin[2]+dbin[3]))*(xlog2x(dbin[1]/float(dbin[1]+dbin[0]))+xlog2x(dbin[0]/float(dbin[1]+dbin[0])))
    cond_ent2 = -((dbin[2]+dbin[3])/float(dbin[1]+dbin[0]+dbin[2]+dbin[3]))*(xlog2x(dbin[3]/float(dbin[3]+dbin[2]))+xlog2x(dbin[2]/float(dbin[3]+dbin[2])))
    return(cond_ent1+cond_ent2)

#print(cond_ent(csvtomatrix("education_train.csv"),1))

#print(xlog2x(4/float(5)))

def at_r_names(data):
    return data[0]




def at_r_val(data):
    val1 = data[1]
    val2 = []
    for i in range(len(data[0])):
        for j in range(len(data)-2):
            if val1[i] != data[j+2][i]:
                val2.append(data[j+2][i])
                break
        if len(val2) == i:
            val2.append(None)
    return [val1,val2]

#'notA', 'notA', 'A', 'notA', 'A', 'A', 'A', 'notA', 'notA', 'A', 'A'
#'A', 'A', 'notA', 'A', 'notA', 'notA', 'notA', 'A', 'A', 'notA', 'notA'

#print(at_r_val(csvtomatrix("a.txt")))

def info_gain(data):
    a = []
    for i in range(len(data[0])-1):
        a.append(entropy(data)-cond_ent(data,i+1))
    return a

#print(info_gain(csvtomatrix("education_train.csv")))
#

def max_array(arr):
    val = 0
    for i in range(len(arr)):
        if arr[i] > arr[val]:
            val = i
    return [val,arr[val]]

#print(max_array([1,2,3,4,7,8,5,6,270.0]))

def subset_rows(data, val, num):
    a_branch =[]
    if num == 0:
        chk = at_r_val(data)[0][val]
    elif num == 1:
        chk = at_r_val(data)[1][val]
    a_branch.append(data[0])
    for i in range(len(data)-1):
        if(data[i+1][val] == chk):
            a_branch.append(data[i+1])
    return a_branch

#print(subset_rows(csvtomatrix("Book2.csv"),0,1))
#
#print(len(subset_rows(csvtomatrix("Book2.csv"),0,1)))



def subset_chop(data, val):
    a1 = []
    for i in range(len(data)):
        a2 = []
        for j in range(len(data[0])):
            if j != val:
                a2.append(data[i][j])
        a1.append(a2)
    return a1

#print(subset_chop(csvtomatrix("Book2.csv"),4))    

#print(subset_chop(subset_rows(csvtomatrix("Book2.csv"),0,1),0))

#print(at_r_val(csvtomatrix("Book2.csv"))[0][len(at_r_val(csvtomatrix("Book2.csv"))[0])-1])
#print(at_r_val(csvtomatrix("Book2.csv"))[1][len(at_r_val(csvtomatrix("Book2.csv"))[0])-1])


def count_0_1(data):
    a0 = data[1][len(data[0])-1]
    a0c = 0
    a1c = 0
    a1 = None
    for i in range(len(data)-1):
        if a0 == data[i+1][len(data[0])-1]:
            a0c+=1
        else:
            a1c+=1
            a1 = data[i+1][len(data[0])-1]
    return [[a0,a1],[a0c,a1c]]
            
def vdep(depth):
    a = ""
    for i in range(depth):
        a = a+"|"
    return a

def pretty_tree(depth, data, att_name):
    a = vdep(depth)
    x = count_0_1(data)
    return(a+att_name+"=["+str(x[1][0])+x[0][0]+"/"+str(x[1][1])+x[0][1]+"]")
        


def list_dict(data):
    a_n = data[0]
    a_0 = at_r_val(data)[0]
    a_1 = at_r_val(data)[1]
    return [a_n,a_0,a_1]
    


def recur(data,root,depth, max_depth):
    a0 = at_r_val(data)[0][len(at_r_val(data)[0])-1]
    a1 = at_r_val(data)[1][len(at_r_val(data)[0])-1]
    a = [0,0]
    if depth == 0:
#        x = count_0_1(data)
        print("["+str(count_0_1(data)[1][0])+count_0_1(data)[0][0]+"/"+str(count_0_1(data)[1][1])+count_0_1(data)[0][1]+"]")
    if depth == max_depth or len(data[0])==1 or max_array(info_gain(data))[1] == 0:
        for i in range(len(data)-1):
            if a0 == data[i+1][len(data[0])-1]:
                a[0]+=1
            else:
                a[1]+=1
        if a[0]>=a[1]:
            root.val = a0
        else:
            root.val = a1
        return
    else:
        info = info_gain(data)
        index = max_array(info)[0]
        # node value
        root.att_name = at_r_names(data)[index]
#        print(node.att_name)
        root.l_br_v = at_r_val(data)[0][index]
        root.r_br_v = at_r_val(data)[1][index]
        
        subset0 = subset_chop(subset_rows(data,index,0),index)
        
        subset1 = subset_chop(subset_rows(data,index,1),index)
        root.set_br(node(),node())
#        node.r_br = node()
        print(pretty_tree(depth+1,subset0,root.att_name))
        recur(subset0,root.l_br,depth+1, max_depth)
        
        print(pretty_tree(depth+1,subset1,root.att_name))
        recur(subset1,root.r_br,depth+1, max_depth)
        return

root = node()

recur(csvtomatrix("education_train.csv"),root,0,3)
        