#!/bin/python

# Contextual bandits tests with TOPKS context vectors

import sys
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from bandits import LinearUCB
from searcher import simulateQueryPairs, generateContexts


N = 300
T = 1000
k = 5
VERBOSE = False
theta = np.array([0.1,0.8])

# Queries simulation
triplesFile = sys.argv[1]
"""
 Get 2 query pairs datasets:
   1. to estimate theta_0 (bias of regression)
   2. to learn
"""
pairs_beta0_est, pairs_learning = simulateQueryPairs(triplesFile, N, T)

# Generate all Contexts
AllContexts = []
for pair in pairs_beta0_est:
    AllContexts.append(generateContexts(pair, theta, k))
n_vecs = sum([len(C_t) for C_t in AllContexts])
theta_0 = - np.dot(sum([sum(Contexts_t) for Contexts_t in AllContexts]), theta) / n_vecs

# Bandit parameters
noise = stats.norm(0, 0.2)
alpha = 0.01
beta = np.append(theta_0, theta)
linUCB = LinearUCB(beta, alpha, noise)

# Estimated cumulated regret up to time T

rewards, arms = [], []
bestRewards, bestArms = [], []

print 'Beta'
print str(beta)

print 'First contexts'

counter = 1
for pair in pairs_learning:
    if counter%100 == 0:
        print counter
    counter += 1
    linUCB.computeEstimation()
    # Create new Contexts
    if linUCB.beta_estimation is None or sum(linUCB.beta_estimation[1:]) == 0:
        Contexts_t = generateContexts(pair, np.array([0.5,0.5]), k)
    else:
        if VERBOSE:
            print "Estimation"
            print str(linUCB.beta_estimation)
        Contexts_t = generateContexts(pair, linUCB.beta_estimation[1:], k)
    Contexts_t = [np.append(1, x) for x in Contexts_t]
    # Use our policy to choose next action
    x, reward = linUCB.play(Contexts_t)
    # Save our action and observation
    arms.append(x)
    rewards.append(reward)
    # Compute Best theoretical value
    bestArm, bestReward = linUCB.getBestReward(Contexts_t)
    # Save theoretical best action and observation
    bestArms.append(bestArm)
    bestRewards.append(bestReward)
    if VERBOSE:
        print str(Contexts_t)
        print "Chosen"
        print str(x)
        print str(reward)
        print "Best"
        print str(bestArm)
        print str(bestReward)
        raw_input("Press Enter to continue...")

sumRewards = np.cumsum(rewards)
sumBestRewards = np.cumsum(bestRewards)

# Plot

regret = (sumBestRewards - sumRewards)
fig = plt.figure(figsize=(7, 6))
plt.plot(np.arange(1,T+1),regret,label='LinUCB')
plt.legend()
plt.savefig('./img/'+'test.png')
plt.close(fig)
