# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:38:19 2018

@author: Yash Kumar
"""

import numpy as np

def dicdata(input_txt):
    f = open(input_txt)
    data = [l.strip() for l in f.readlines()]
    dicword = {}
    for i,v in enumerate(data):
        dicword[v] = i
    return dicword, data

def trainmatrix(input_txt):
    f = open(input_txt)
    data = [l.strip() for l in f.readlines()]
    a = []
    for i in range(len(data)):
        temp1 = data[i].split(" ")
        a.append(temp1)
    return a

def split_stt(train_data):
    return [tuple(a.split("_"))[0] for a in train_data]

def split_lbl(train_data):
    return [tuple(a.split("_"))[1] for a in train_data]        

dic_labels, llabels = dicdata("index_to_tag.txt")
dic_states, _ = dicdata("index_to_word.txt")

train_data = trainmatrix("trainwords.txt")
train_states = list(map(split_stt, train_data))
train_labels = list(map(split_lbl, train_data))
        
test_data = trainmatrix("testwords.txt")
test_states = list(map(split_stt, test_data))
test_labels = list(map(split_lbl, test_data))

in_prob = np.zeros(len(dic_labels))
tran_prob = np.zeros((len(dic_labels),len(dic_labels)))
em_prob = np.zeros((len(dic_labels),len(dic_states)))

for i in range(len(train_labels)):
    in_prob[dic_labels[train_labels[i][0]]]+=1
    
in_prob = (in_prob+1)/(sum(in_prob)+len(in_prob))

for i in range(len(train_labels)):
    for j in range(len(train_labels[i])-1):
        tran_prob[dic_labels[train_labels[i][j+1]],dic_labels[train_labels[i][j]]]+=1
#tran_prob = tran_prob.T

for i in range(len(train_labels[0])):
    tran_prob[:,i] = (tran_prob[:,i]+1)/(sum(tran_prob[:,i])+len(dic_labels))
tran_prob = tran_prob.T


for i in range(len(train_labels)):
    for j in range(len(train_labels[i])):
        em_prob[dic_labels[train_labels[i][j]],dic_states[train_states[i][j]]]+=1


for i in range(len(dic_labels)):
    em_prob[i,:] = (em_prob[i,:]+1)/(sum(em_prob[i,:])+len(dic_states))



        
def alpha(j, t, sentence, dic_a):
    if (j, t) in dic_a:
        return dic_a[(j, t)]
    elif t == 0:
        dic_a[(j, t)] = in_prob[j]*em_prob[j,dic_states[sentence[0]]]
        return dic_a[(j, t)]
    else:
        newsum = 0
        for k in range(len(dic_labels)):
            if (k, t-1) in dic_a:
                newsum += tran_prob[k,j]*dic_a[(k, t-1)]
            else:
                dic_a[(k, t-1)] = alpha(k, t-1, sentence, dic_a)
                newsum += tran_prob[k,j]*dic_a[(k, t-1)]
        return newsum*em_prob[j,dic_states[sentence[t]]]

for qq in range(9):
    dic_a = {}
    print(alpha(qq, 3, ['EU', '*OOV*', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], dic_a))

print(dic_a)

#def beta(j, t, sentence):
#    if t == len(sentence)-1:
#        return 1
#    else:
#        newsum = 0
#        for k in range(len(dic_labels)):
#            newsum += em_prob[k, dic_states[sentence[t+1]]]*beta(k, t+1, sentence)*tran_prob[j,k]
#        return newsum
#
#def alphabeta(j, t, sentence):
#    dic_a = {}
#    dic_b = {}
#    return alpha(j, t, sentence, dic_a)*beta(j, t, sentence, dic_b)



#def argmax(t, sentence):
#    args = []
#    for j in range(len(dic_labels)):
#        args.append(alphabeta(j, t, sentence))
#    return llabels[max(enumerate(args))[0]]
        
#for a in range(9):
#    print(argmax(a, ['Only', 'France', 'and', 'Britain', 'backed', 'Fischler', "'s", 'proposal', '.']))
