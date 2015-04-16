#!/usr/bin/env python
#-*-coding: utf-8 -*-

"""
Analysis on the social and textual scores
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from searcher import getContexts

sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (12, 8)})


N = 1000
MIN_TAG = 5
NB_QUERIES = 100

def create_queries(nb_queries, filename, min_tag):
    """
    Given triples (input file with name filename), this function creates nb_queries queries
    whose tag is used at least min_tag times.
    """
    # Create a dictionary whose values are the number of times each tag was used
    tags = dict()
    with open(filename, 'r') as f:
        for line in f:
            user, item, tag = line.rstrip('\n').split('\t')
            user = int(user)
            item = int(item)
            if tag not in tags:
                tags[tag] = 0
            tags[tag] += 1
    print 'Number of tags: ', str(len(tags)), ' ...'
    # Extract good tags from the dictionary
    good_tags = set()
    for tag in tags:
        if tags[tag] >= min_tag:
            good_tags.add(tag)
    print 'Number of good tags: ', str(len(good_tags)), ' ...'
    # Create nb_queries respecting the rules from above
    queries = list()
    with open(filename, 'r') as f:
        counter = 0
        for line in f:
            counter += 1
            if (counter+1) % 1000 != 0:
                continue
            user, item, tag = line.rstrip('\n').split('\t')
            user = int(user)
            item = int(item)
            if tag in good_tags:
                queries.append((user,tag))
            if len(queries) >= nb_queries:
                break
    print 'Number of queries: %d' % len(queries)
    print queries[:4]
    return queries

def statistics_scores(queries, n):
    """
    This function returns the mean of textual and social scores at the top-n positions when
    typing the queries given as input.

    Parameters:
    -----------
    queries: list of tuples
        (seeker, keyword)
    n: int
        top-n answer
    """
    mean_textual_ranking = np.zeros(n)
    mean_social_ranking = np.zeros(n)
    seen = np.zeros(N)
    for query in queries:
        seeker, keyword = query
        Contexts = getContexts(seeker, keyword, 0.1, N)
        textual = sorted([x[0] for x in Contexts], reverse=True)
        social = sorted([x[1] for x in Contexts], reverse=True)
        for i in xrange(len(Contexts)):
            mean_textual_ranking[i] += textual[i]
            mean_social_ranking[i] += social[i]
            seen[i] += 1
    for i in xrange(len(seen)):
        if seen[i] == 0:
            break
        mean_textual_ranking[i] /= seen[i]
        mean_social_ranking[i] /= seen[i]
    return mean_textual_ranking, mean_social_ranking

def plot_mean_ranking(textual, social, filename):
    """
    Plot the evolution of textual and social score when increasing the ranking
    """
    fig = plt.figure(figsize=(7, 6))
    t = np.arange(1, len(textual)+1)
    tex, = plt.plot(t, textual, 'r^', label='Textual')
    soc, = plt.plot(t, social, 'bs', label='Social')
    plt.title('Average scores at each ranking')
    plt.legend()
    plt.savefig('./img/'+filename+'.png')
    plt.close(fig)
    return

if __name__ == '__main__':
    filename = sys.argv[1]
    queries = create_queries(NB_QUERIES, filename, MIN_TAG)
    textual, social = statistics_scores(queries, N)
    plot_mean_ranking(textual, social, 'scores_analysis')
