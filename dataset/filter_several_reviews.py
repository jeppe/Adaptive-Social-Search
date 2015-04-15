#!/usr/bin/env python
#-*-coding: utf-8 -*-

import sys

MIN_REVIEWS = 5

items = dict()
for line in open(sys.argv[1], 'r'):
    u, i, t = line.rstrip('\n').split('\t')
    u = int(u)
    i = int(i)
    if i not in items:
        items[i] = set()
    items[i].add(u)

good_items = set()
for i in items:
    if len(items[i]) >= MIN_REVIEWS:
        good_items.add(i)

#print len(good_items)

for line in open(sys.argv[1], 'r'):
    u, i, t = line.rstrip('\n').split('\t')
    u = int(u)
    i = int(i)
    if i not in good_items:
        continue
    print "%d\t%d\t%s" % (u, i, t)
