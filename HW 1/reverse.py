# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input")

parser.add_argument("output")
args = parser.parse_args()



with open (args.input,'r') as line:
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

with open(args.output,'w') as file:
    file.writelines(a)
    