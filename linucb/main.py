#!/bin/python

# Contextual bandits tests with TOPKS context vectors

import sys
import numpy as np
from numpy.linalg import norm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from bandits import LinearUCB
from searcher import simulateQueryPairs, generateContexts


N = 300
T = 10000
k = 4
VERBOSE = False
theta = np.array([0.3,0.7])

# Queries simulation
triplesFile = sys.argv[1]
"""
 Get 2 query pairs datasets:
   1. to estimate theta_0 (bias of regression)
   2. to learn
"""
#pairs_beta0_est, pairs_learning = simulateQueryPairs(triplesFile, N, T)

# Generate all Contexts
#AllContexts = []
#for pair in pairs_beta0_est:
#    curr_context = generateContexts(pair, theta, k)
#    if len(curr_context) < k:
#        continue
#    AllContexts.append(curr_context)
AllContexts = [np.array([1,1]), np.array([1.5,1]), np.array([1.25,1]), np.array([1,1.25]), np.array([1,1.5])]
AllContexts = map(lambda x: x/norm(x), AllContexts)

print 'Total number of contexts:', str(len(AllContexts))
#n_vecs = sum([len(C_t) for C_t in AllContexts])
#theta_0 = - np.dot(sum([sum(Contexts_t) for Contexts_t in AllContexts]), theta) / n_vecs
#theta_0 = -np.dot(sum(AllContexts), theta) / 5
theta_0 = -np.dot(AllContexts[4],theta)

# Bandit parameters
noise = stats.norm(0, 0.3)
alpha = 1.0
beta = np.append(theta_0, theta)

# Debug on Contexts
newAllContexts = []
for x in AllContexts:
    newAllContexts.append(np.append(1, x))
AllContexts = newAllContexts

for x in AllContexts:
    print str(x), ' : ', str(np.dot(x, beta))

linUCB = LinearUCB(beta, alpha, noise)

# Estimated cumulated regret up to time T

rewards, arms = [], []
bestRewards, bestArms = [], []

print 'Beta'
print str(beta)

linUCB.computeEstimation()

print 'First contexts'

counter = 1
result = {0:0, 1:0, 2:0, 3:0, 4:0}
#for pair in pairs_learning:
for i in range(T):
    if counter%100 == 0:
        print counter
    # Create new Contexts
    #if linUCB.beta_estimation is None or sum(linUCB.beta_estimation[1:]) == 0:
    #    Contexts_t = generateContexts(pair, np.array([0.5,0.5]), k)
    #else:
    #    if VERBOSE:
    #        print "Estimation"
    #        print str(linUCB.beta_estimation)
    #    Contexts_t = generateContexts(pair, linUCB.beta_estimation[1:], k)
    #if len(Contexts_t) < k:
    #    continue
    Contexts_t = AllContexts
    counter += 1
    #Contexts_t = [np.append(1, x) for x in Contexts_t]
    # Use our policy to choose next action
    arm, reward = linUCB.play(Contexts_t)
    # Save our action and observation
    arms.append(arm)
    result[arm] += reward
    rewards.append(reward)
    # Compute Best theoretical value
    bestArm, bestReward = linUCB.getBestReward(Contexts_t)
    # Save theoretical best action and observation
    bestArms.append(bestArm)
    bestRewards.append(bestReward)
    linUCB.computeEstimation()
    if VERBOSE:
        print str(Contexts_t)
        print "Chosen"
        print str(x)
        print str(reward)
        print "Best"
        print str(bestArm)
        print str(bestReward)
        print "Beta estimation"
        print str(linUCB.beta_estimation)
        print "B"
        print str(linUCB.B)
        print "b"
        print str(linUCB.b)
        raw_input("Press Enter to continue...")

from collections import Counter
print Counter(arms)
print result

sumRewards = np.cumsum(rewards)
sumBestRewards = np.cumsum(bestRewards)

# Plots

regret = (sumBestRewards - sumRewards)
fig = plt.figure(figsize=(7, 6))
plt.plot(np.arange(1,len(regret)+1),regret,label='LinUCB')
plt.legend()
plt.savefig('./img/'+'test.png')
plt.close(fig)

print linUCB.beta
print linUCB.beta_estimation
