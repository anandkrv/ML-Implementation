# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:59:23 2018

@author: Yash Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:30:48 2018

@author: Yash Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:03:40 2018

@author: Yash Kumar
"""

import numpy as np
import copy

#train input
#test input
#max depth
#train out
#test out
#metrics out

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("traininput")
parser.add_argument("testinput")
parser.add_argument("maxdepth")
parser.add_argument("trainoutput")
parser.add_argument("testoutput")
parser.add_argument("metricsout")

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

class node(object):
    def __init__(self):
        self.att_name = None
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
    if dbin[2]+dbin[3]==0 or dbin[0]+dbin[1]==0:
        return entropy(data)
    else:
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

def subset_rows(datacopy, val, acheck):
    data = copy.deepcopy(datacopy)
    a_branch =[]
    a_branch.append(data[0])
    for i in range(len(data)-1):
        if(data[i+1][val] == acheck):
            a_branch.append(data[i+1])
    return a_branch

#print(subset_rows(csvtomatrix("Book2.csv"),0,1))
#
#print(len(subset_rows(csvtomatrix("Book2.csv"),0,1)))



def subset_chop(datacopy, val):
    data = copy.deepcopy(datacopy)
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


def count_0_1(data, att_list):
    a0 = att_list[1][len(att_list[0])-1]
    a1 = att_list[2][len(att_list[0])-1]
    a0c = 0
    a1c = 0
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

def pretty_tree(depth, data, att_name, att_val, att_list):
    a = vdep(depth)
    a0 = att_list[1][len(att_list[0])-1]
    a1 = att_list[2][len(att_list[0])-1]
    x = count(data,a0,a1)
    return(a+att_name+"="+att_val+"["+str(x[0])+a0+"/"+str(x[1])+a1+"]")
        


def list_dict(data):
    a_n = data[0]
    a_0 = at_r_val(data)[0]
    a_1 = at_r_val(data)[1]
    a_2 = []
    a_3 = []
    for i in range(len(data[0])):
        a_2.append(1)
    for i in range(len(data[0])):
        a_3.append(i)
    return [a_n,a_0,a_1,a_2, a_3]
    

#print(list_dict(csvtomatrix("politicians_train.csv")))


def count(data, a0, a1):
    a0c = 0
    a1c = 0
    for i in range(len(data)-1):
        if data[i+1][len(data[0])-1] == a0:
            a0c+=1
        else:
            a1c+=1
    return[a0c,a1c]

# change in info_gain2 required 
# conditional entropy gain is not able to find out the
# 
            
def info_gain2(data, att_list):
    b1 = []
    b2 = []
    b3 = []
    j = 0
    for i in range(len(att_list[0])-1):
        if att_list[3][i] != 0:
            b1.append(entropy(data)-cond_ent(data,j+1))
            b2.append(att_list[0][i])
            b3.append(att_list[4][i])
            j+=1
        
#            i=1
#            b1.append(None)
#            b2.append(None)
#            b3.append(None)
    return [b1,b2,b3]

def max_array2(b):
    max_ele = 0
    max_ind = -1
    for i in range(len(b[0])):
        if b[0][i]!=None:
            if max_ele < b[0][i]:
                max_ele = b[0][i]
                max_ind = b[2][i]
    return [max_ind, max_ele]
            

def act_index(data, att_name):
#    print(len(data[0]))
    for i in range(len(data[0])):
#        print(data[0][i])
        if att_name == data[0][i]:
            return i
    return None

#print(csvtomatrix("politicians_train.csv")[0])
#print(act_index(csvtomatrix("politicians_test.csv"),"F"))

def attlistupdate(att_list, index):
    newattlist = copy.deepcopy(att_list)
    newattlist[3][index] = 0
    return newattlist

def recur(root, depth, maxdepth, data, att_list):
    if depth == 0:
        att_list = list_dict(data)
#        print(att_list)
    a0 = att_list[1][len(att_list[0])-1]
    a1 = att_list[2][len(att_list[0])-1]
    if depth == 0:
        print("["+str(count(data,a0,a1)[0])+a0+"/"+str(count(data,a0,a1)[1])+a1+"]")
    
    if depth == maxdepth or len(data[0]) == 1 or max_array2(info_gain2(data, att_list))[0] == -1:
#        print(info_gain2(data, att_list))
#        print(entropy(data))
        a0c = count(data,a0,a1)[0]
        a1c = count(data,a0,a1)[1]
        if a0c >= a1c:
            root.val = a0
        else:
            root.val = a1
        return
    else:
#        print(info_gain2(data, att_list))
#        print(entropy(data))
#        print(info_gain2(data, att_list))
        index = max_array2(info_gain2(data, att_list))[0]
        root.att_name = att_list[0][index]
        root.l_br_v = att_list[1][index]
        root.r_br_v = att_list[2][index]
#        print(root.att_name)
# remember indexcheck is not taking into account whether l_branch_value or r_branch_value will be selected
        indexcheck = act_index(data, root.att_name)
        subset0 = subset_chop(subset_rows(data,indexcheck,root.l_br_v),indexcheck)
        subset1 = subset_chop(subset_rows(data,indexcheck,root.r_br_v),indexcheck)
        attlist1 = attlistupdate(att_list, index)
        attlist2 = attlistupdate(att_list, index)
        root.set_br(node(),node())
        print(pretty_tree(depth+1,subset0,root.att_name,root.l_br_v,att_list))
#        print() 10:12AM
#        print(attlist1[0],attlist1[3])
        recur(root.l_br,depth+1, maxdepth,subset0, attlist1)
        
        print(pretty_tree(depth+1,subset1,root.att_name,root.r_br_v,att_list))
#        print(attlist1[0],attlist1[3])
        recur(root.r_br,depth+1, maxdepth,subset1, attlist2)
        return
    
root = node()

recur(root,0,3,csvtomatrix(args.traininput),[])

def search(name, data):
    for i in range(len(data)):
        if data[i] == name:
#            print(data[i])
            return i

def err_output_write(root, data, out_txt):
#    root = check
    correct = 0
    wrong = 0
    with open(out_txt,'w') as f:
        for i in range(len(data)-1):
            chk = root
            while chk.val==None:
                index = search(chk.att_name,data[0])
                if data[i+1][index] == chk.l_br_v:
                    chk = chk.l_br
                else:
    #            elif data[i+1][index] == chk.r_br_v:
                    chk = chk.r_br
#            print(chk.val)
            f.write(chk.val+"\n")
#            f.write("\n")
            if chk.val == data[i+1][len(data[0])-1]:
                correct+=1
            else:
                wrong+=1
    return (wrong/float(correct+wrong))

trainerr = err_output_write(root,csvtomatrix(args.traininput), args.trainoutput)
#print(trainerr)

testerr = err_output_write(root,csvtomatrix(args.testinput), args.testoutput)
#
#print(trainerr,testerr)

def write_errfile(trainerr, testerr, errfile):
    with open(errfile,'w') as f:
        f.write("error(train): "+str(trainerr)+"\n"+"error(test): "+str(testerr))
#        f.write(str(trainerr))
#        f.write("\n")
#        f.write("error(test): ")
#        f.write(str(testerr))
    return

write_errfile(trainerr, testerr, args.metricsout)


#def what_is(root):    
#    chk = root
#    while(chk != None):
#        print(chk.att_name)
#        print(chk.val)
#        print("left")
#        print(chk.l_br_v)
#        what_is(chk.l_br)
#        print("right")
#        print(chk.r_br_v)
#        what_is(chk.r_br)
#        return
#    return
#
#what_is(root)