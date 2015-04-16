#!/bin/python

import urllib2
import json
import random
import numpy as np
import sys

# Horizon
DOMAIN = 'http://localhost:8000/topks'  # Domain of the TOPKS search server
TIME = 5000                             # No limit, we want to visit the whole graph
N_NEIGH = 100000                        # Whole graph
NEW_QUERY = 'true'                      # new query every time (no prefix strategy)
ALPHA = 0                               # search with alpha in the balance 


def getContexts(seeker, query, alpha, k):
    """
    Given a seeker and a query, send a GET http request to the TOPKS
    server. Parse the JSON reponse and return context vectors

    Parameters:
    -----------
    seeker: int
    query: list of String
    beta: numpy array (1*d)
    k: int
    """
    Contexts = []
    url = DOMAIN + \
            '?q=' + '+'.join(query.split()) + \
            '&seeker=' + str(seeker) + \
            '&t=' + str(TIME) + \
            '&newQuery=' + NEW_QUERY + \
            '&nNeigh=' + str(N_NEIGH) + \
            '&alpha=' + str(alpha) + \
            '&k='+str(k)
    try:
        response = urllib2.urlopen(url)
        data = json.load(response)
        if not data.has_key('status') or not data.has_key('results'):
            return None
        if data.get('status') != 1:
            return None
        results = data.get('results')
        for x in results:
            Contexts.append(np.array([x.get('textualScore'), x.get('socialScore')]))
    except urllib2.HTTPError, error:
        print error.read()
    except (ValueError, KeyError, TypeError) as error:
        print error
    return Contexts

def generateContexts(queryPair, betaEst, k):
    alpha = betaEst[0] / sum(betaEst)
    Contexts = getContexts(queryPair[0], queryPair[1], alpha, k)
    return Contexts

def simulateQueryPairs(triplesFile, N, T):
    users, tags = set(), set()
    pairs = []
    with open(triplesFile, 'r') as f:
        for line in f:
            data = line.rstrip('\n').split('\t')
            if len(data) != 4:
                continue
            u = data[0]
            i = data[1]
            t = random.choice(data[2].split(','))
            pairs.append((u,[t]))
    print str(pairs[:3])
    return pairs[:N], pairs[N:N+T]
            #users.add(u)
            #tags.add(t)
        #users = list(users)
        #tags = list(tags)
    
    #user_indexes_est = np.random.randint(len(users), size=N)
    #tag_indexes_est = np.random.randint(len(tags), size=N)
    #pairs_beta0_est = []
    #for i, j in zip(user_indexes_est, tag_indexes_est):
    #    pairs_beta0_est.append((users[i], [tags[j]]))
    #
    #user_indexes_lrng = np.random.randint(len(users), size=T)
    #tag_indexes_lrng = np.random.randint(len(tags), size=T)
    #pairs_learning = []
    #for i, j in zip(user_indexes_lrng, tag_indexes_lrng):
    #    pairs_learning.append((users[i], [tags[j]]))
 
    #return pairs_beta0_est, pairs_learning
