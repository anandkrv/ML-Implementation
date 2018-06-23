# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



with open ('input.txt','r') as line:
    temp = line.readlines()

print(temp)

#temp[0] = temp[0].replace('\n','')
#print(temp)

a = []




for i in range(len(temp)-1,-1,-1):
#    if i == len(temp)-1:
#        a.append(temp[i] + '\n')
#    else:
        a.append(temp[i])

print(a)

with open('output.txt','w') as file:
    file.writelines(a)
    