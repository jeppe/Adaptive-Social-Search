#!/usr/bin/env python
#-*-coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import fileinput

"""
Analysis on Users and friends
"""

sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (12, 8)})

MIN_FRIENDS=5

DF = pd.DataFrame(json.loads(l) for l in fileinput.input())

good_users = {}
counter = 1
for i in range(len(DF)):
    if len(DF.iloc[i]['friends']) >= MIN_FRIENDS:
        good_users[DF.iloc[i]['user_id']] = ([], counter)
        counter += 1
print len(good_users)

for i in range(len(DF)):
    uid = DF.iloc[i]['user_id']
    if uid in good_users:
        for f in DF.iloc[i]['friends']:
            if f in good_users:
                good_users[uid][0].append(f)

with open('social_network.txt', 'w') as f:
    for i in range(len(DF)):
        uid = DF.iloc[i]['user_id']
        if uid in good_users:
            for friend in good_users[uid][0]:
                f.write( '%d\t%d\n' % (good_users[uid][1], good_users[friend][1]) )

data = map(len, map(lambda x: x[0], good_users.values()))

with open('good_users.txt', 'w') as outputFile:
    for item in good_users:
        outputFile.write( '%s\t%d\n' % (item, good_users[item][1]) )

bins = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100,200,500,1000,10000]

bin_middles = bins[:-1] + np.diff(bins)/2.
bar_width = 1. 
m, bins = np.histogram(data, bins)
plt.bar(np.arange(len(m)) + (1-bar_width)/2., m, width=bar_width)
plt.title('Yelp number of friends distribution')
ax = plt.gca()
ax.set_xticks(np.arange(len(bins)))
ax.set_xticklabels(['{:.0f}'.format(i) for i in bins])

plt.show()
