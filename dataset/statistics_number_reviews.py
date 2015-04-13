#!/usr/bin/env python
#-*-coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import fileinput

"""
Analysis on Reviews
"""

sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (12, 8)})

DF = pd.DataFrame(json.loads(l) for l in fileinput.input())

good_users = {}
with open('good_users.txt', 'r') as f:
    for l in f:
        l = l.rstrip('\n')
        hash_id, long_id = l.split('\t')
        good_users[hash_id] = (set(), long_id)

filtered_DF = DF[DF['user_id'].isin(good_users.keys())]
filtered_DF.to_json('good_reviews')
#    for i in range(len(DF)):
#        if (i+1)%100000==0:
#            print i+1
#        id_ = DF.iloc[i]['user_id']
#        if id_ not in good_users:
#            continue
#        good_users[id_][0].add(DF.iloc[i]['review_id'])
#        f.write(
#
#print len(good_users)
#
#counter = 0
#for user in good_users:
#    if len(good_users[user][0]) == 0:
#        counter += 1 #print 'zero'
#print counter
#
#data = map(len, map(lambda x: x[0], good_users.values()))
#bins = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100,200,500,1000,10000]
#
#
#bin_middles = bins[:-1] + np.diff(bins)/2.
#bar_width = 1. 
#m, bins = np.histogram(data, bins)
#plt.bar(np.arange(len(m)) + (1-bar_width)/2., m, width=bar_width)
#plt.title('Yelp number of reviews distribution (per user)')
#ax = plt.gca()
#ax.set_xticks(np.arange(len(bins)))
#ax.set_xticklabels(['{:.0f}'.format(i) for i in bins])
#
#plt.show()
