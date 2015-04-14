#!/bin/python

"""
Contextual bandits tests with TOPKS context vectors
"""
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
T = 1000000
k = 4
VERBOSE = False
#theta = np.array([0.3,0.7])

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
theta = (2*AllContexts[1]+3*AllContexts[2])/5

print 'Total number of contexts:', str(len(AllContexts))
#n_vecs = sum([len(C_t) for C_t in AllContexts])
#theta_0 = - np.dot(sum([sum(Contexts_t) for Contexts_t in AllContexts]), theta) / n_vecs
#theta_0 = -np.dot(sum(AllContexts), theta) / 5
#theta_0 = -np.dot(AllContexts[1],theta)

# Bandit parameters
noise = np.random.normal(0,.1,T)
alpha = 1.
#beta = np.append(theta_0, theta)
beta = theta

# Debug on Contexts
#newAllContexts = []
#for x in AllContexts:
#    newAllContexts.append(np.append(0, x))
#AllContexts = newAllContexts

for x in AllContexts:
    print str(x), ' : ', str(np.dot(x, beta))

linUCB = LinearUCB(beta, alpha, noise)

# Estimated cumulated regret up to time T

rewards, arms = [], []
bestRewards, bestArms = [], []
betaEstimations = []

print 'Beta'
print str(beta)

linUCB.computeEstimation()

print 'First contexts'

counter = 1
result = {0:0, 1:0, 2:0, 3:0, 4:0}
#for pair in pairs_learning:
for i in range(T):
    if (counter+1)%1000 == 0:
        print (counter+1)
    betaEstimations.append(norm(linUCB.beta-linUCB.beta_estimation))
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
#print result

sumRewards = np.cumsum(rewards)
sumBestRewards = np.cumsum(bestRewards)

# Plots

regret = (sumBestRewards - sumRewards)
fig = plt.figure(figsize=(7, 6))
plt.plot(np.arange(1,len(regret)+1),regret,label='LinUCB')
plt.legend()
plt.savefig('./img/'+'test.png')
plt.close(fig)

fig = plt.figure(figsize=(7, 6))
plt.axis([0, T, 0, 0.3])
plt.plot(np.arange(1,len(betaEstimations)+1),betaEstimations,label='loss distance distance')
plt.legend()
plt.savefig('./img/'+'beta_estimation.png')
plt.close(fig)

fig = plt.gcf()#figure(figsize=(7, 6))
plt.gca().set_xlim((-1.2,1.2))
plt.gca().set_ylim((-1.2,1.2))
plt.gca().plot(np.array(map(lambda x: x[0],AllContexts)),np.array(map(lambda x: x[1],AllContexts)),'o',color='black')
plt.gca().plot(theta[0], theta[1],'o',color='blue')
normalisation = norm(np.array([linUCB.beta_estimation[0],linUCB.beta_estimation[1]]))
plt.gca().plot(linUCB.beta_estimation[0]/normalisation, linUCB.beta_estimation[1]/normalisation,'o',color='red')
fig.gca().add_artist(plt.Circle((0,0),1.,color='b',fill=False))
for i, x in enumerate(AllContexts):
    fig.gca().annotate('%d' % i, xy=(x[0],x[1]), xytext=(x[0], x[1]),
                       arrowprops=dict(facecolor='black', shrink=0.05),
                      )
fig.savefig('img/points.png')

print linUCB.beta
print linUCB.beta_estimation
