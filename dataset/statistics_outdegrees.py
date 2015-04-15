#!/usr/bin/env python
#-*-coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
import fileinput

"""
Analysis on Users and friends
"""

#sns.set_palette("deep", desat=.6)
#sns.set_context(rc={"figure.figsize": (12, 8)})

users = dict()
for l in fileinput.input():
    u1, u2 = map(int, l.rstrip('\n').split('\t')[:2])
    if u1 not in users:
        users[u1] = set()
    if u2 not in users:
        users[u2] = set()
    users[u1].add(u2)
    users[u2].add(u1)

data = map(len, users.values())

#print len(data)
data += [0] * (77272 - len(data))

bins = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100,200,500,1000,10000]

bin_middles = bins[:-1] + np.diff(bins)/2.
bar_width = 1. 
m, bins = np.histogram(data, bins)

fig = plt.figure(figsize=(7, 6))
plt.bar(np.arange(len(m)) + (1-bar_width)/2., m, width=bar_width)
plt.title('Yelp social network similarity outdegree distribution')
ax = plt.gca()
ax.set_xticks(np.arange(len(bins)))
ax.set_xticklabels(['{:.0f}'.format(i) for i in bins])
plt.savefig('./img/socialsimoutdegree.png')
#plt.show()
