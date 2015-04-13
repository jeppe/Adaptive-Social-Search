#!/usr/bin/env python
#-*-coding: utf-8 -*-

import json
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from nlp import extract_tags


DF = pd.read_json(sys.argv[1])

# We load good users here
good_users = dict()
with open('good_users.txt', 'r') as f:
    for l in f:
        l = l.rstrip('\n')
        hashid, userid = l.split('\t')
        good_users[hashid] = userid

mapping_reviews = dict()
counter = 1
# We read the reviews, make analysis
with io.open('triples.txt', 'w', encoding='utf8') as outFile:
    for i in range(len(DF)):
        if (i+1) % 50000 == 0:
            print i+1
        user = DF.iloc[i]['user_id']
        item = DF.iloc[i]['business_id']
        if item not in mapping_reviews:
            mapping_reviews[item] = [counter, 0]
            counter += 1
        mapping_reviews[item][1] += 1
        review = DF.iloc[i]['text']
        triples = extract_tags(review)
        for w in triples:
            outFile.write(good_users[user]+'\t'+str(mapping_reviews[item][0])+'\t'+w+'\n')

"""
Plot the number of reviews per item (restaurant, doctor, ...)
"""
data = map(lambda x: x[1], mapping_reviews.values())

bins = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100,200,500,1000,10000]

bin_middles = bins[:-1] + np.diff(bins)/2.
bar_width = 1. 
m, bins = np.histogram(data, bins)
plt.bar(np.arange(len(m)) + (1-bar_width)/2., m, width=bar_width)
plt.title('Yelp number of reviews distribution (per item)')
ax = plt.gca()
ax.set_xticks(np.arange(len(bins)))
ax.set_xticklabels(['{:.0f}'.format(i) for i in bins])

plt.show()
