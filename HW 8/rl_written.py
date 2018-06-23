# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:54:35 2018

@author: Yash Kumar
"""

import operator

q = {}
q2 = 0.8
q[(1, 3)] = 0.6
q[(1, 7)] = -0.3
q[(1, 8)] = -0.5
#q1[(2, 8)] = 0.8
#print(max(q.items(), key = operator.itemgetter(1))[0]) #max(stats.items(), key=operator.itemgetter(1))[0]


r = {}
r[(1, 3)] = 0
r[(1, 7)] = 1
r[(1, 8)] = -1
r2 = 0

for i in range(10000):
    Key = max(q.items(), key = operator.itemgetter(1))[0]
    if Key != (1, 3):
        q[Key] = q[Key] + 0.01*(r[Key] - q[Key])
    else:
        q[(1, 3)] = q[(1, 3)] + 0.01*(r[(1, 3)] - q[(1, 3)])
        q2 = q2 + 0.01*(r2 - q2)
#    q[(1, 7)] = q[(1, 7)] + 0.01*(1 + 0 - q[(1, 7)])
#    q[(1, 8)] = q[(1, 8)] + 0.01*(-1 + 0 - q[(1, 8)])
#    q[(1, 3)] = q[(1, 3)] + 0.01*(0 + 0 - q[(1, 3)])
#    q[(2, 8)] = q[(2, 8)] + 0.01*(0 + 0 - q[(2, 8)])
#

q = {}
q2 = 0.8
q[(1, 3)] = 0.6
q[(1, 7)] = -0.3
q[(1, 8)] = -0.5
#q1[(2, 8)] = 0.8
#print(max(q.items(), key = operator.itemgetter(1))[0]) #max(stats.items(), key=operator.itemgetter(1))[0]


r = {}
r[(1, 3)] = 0
r[(1, 7)] = 1
r[(1, 8)] = -1
r2 = 0

newnew = {key: q[key] - r.get(key, 0) for key in q.keys()}

print(all(abs(value) >= 0.001 for value in newnew.values()))

#print(dict(set(q.items()) - set(r.items())))

#all(map( q.pop, r))
#print(all(map( q.pop, r)))

#{k:v for k,v in A.items() if k not in B}
xxx = []
xxx.append(1)