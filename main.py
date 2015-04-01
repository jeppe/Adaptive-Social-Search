#!/bin/python

# Contextual bandits tests with TOPKS context vectors

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import urllib2
import json

from bandits import LinUCB

# Horizon
T = 1000
DOMAIN = 'http://localhost:8000/topks' # Domain of the TOPKS search server
TIME = 200
N_NEIGH = 100
NEW_QUERY = 'true'
ALPHA = 0


def getContexts(seeker, query, beta, k):
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
            '?q=' + '+'.join(query) + \
            '&seeker=' + str(seeker) + \
            '&t=' + str(TIME) + \
            '&newQuery=' + NEW_QUERY + \
            '&nNeigh=' + str(N_NEIGH) + \
            '&alpha=' + str(ALPHA) + \
            '&k='+str(k)
    response = urllib2.urlopen(url)
    data = json.load(response)
    if not data.has_key('status') or not data.has_key('results'):
        return None
    if data.get('status') != 1:
        return None
    results = data.get('results')
    for x in results:
        Contexts.append(np.array([x.get('textualScore'), x.get('socialScore')]))
    print Contexts
    return Contexts

def generateContexts(T):
    AllContexts = []
    for i in range(T):
        AllContexts.append([])
    return AllContexts

noise = stats.norm(0, 0.2)
alpha = 0.1
beta = np.array([0.5, 0.5])
linUCB = LinUCB(beta, alpha, noise)

AllContexts = generateContexts(T)

# Estimated cumulated regret up to time T

rewards, arms = [], []
bestRewards = []
for Contexts_t in range(AllContexts):
    # Use our policy to choose next action
    x, reward = linUCB.play(Contexts_t)
    # Save our action and observation
    arms.append(x)
    rewards.append(reward)
    # Compute Best theoretical value
    bestReward = linUCB.getBestReward(Contexts_t)
    # Save theoretical best action and observation
    bestArms.append(bestArm)
    bestRewards.append(bestReward)

sumRewards = np.cumsum(rewards)
sumBestRewards = np.cumsum(bestRewards)

#counter = 0
#
#for i in range(N):
#    dumb, rew1 = bd.thompsonSampling(n,MAB)
#    # complete EXP3Stochastic 
#    dumb, rew2 = bd.EXP3Stochastic(n,beta,eta,MAB)
#    reg1 += np.arange(1,n+1)*mumax - np.cumsum(rew1)  # ??? +1  ???
#    reg2 += (1:n)*mumax - np.cumsum(rew2)
#    counter += 1
#    # np.cumsum(np.ones(n)*max(means))-np.cumsum(rew2)
#
#reg1 /= N
#reg2 /= N

plt.plot(np.arange(1,n+1),reg1,label='Thomson Sampling')
plt.plot(np.arange(1,n+1),reg2,label='EXP3')
plt.legend()
plt.show()
