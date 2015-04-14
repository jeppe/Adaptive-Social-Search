#!/usr/bin/env python
#-*-coding: utf-8 -*-

"""
Contextual bandits debug tests (normalised context vectors)
"""
import sys
import numpy as np
from numpy.linalg import norm
from scipy import stats
from collections import Counter

from bandits import LinearUCB
from plot_utils import *


T = 2000000  # number of epochs
Contexts = [np.array([1,1]), np.array([1.5,1]), np.array([1.25,1]),
               np.array([1,1.25]), np.array([1,1.5])]
Contexts = map(lambda x: x/norm(x), Contexts)
theta = (2*Contexts[1] + 3*Contexts[2]) / 5


def experiment(beta, alpha, noise, AllContexts, n_plot=0):
    print 'Total number of contexts:', str(len(AllContexts))
    for x in AllContexts:
        print str(x), ' : ', str(np.dot(x, beta))
    rewards, arms = [], []
    bestRewards, bestArms = [], []
    betaEstimations = []
    counter = 1
    print 'Beta: ', str(beta)
    print 'First contexts'

    linUCB = LinearUCB(beta, alpha, noise)
    linUCB.computeEstimation()          # Compute first beta estimation (identity * 0)
    for i in range(T):
        if (counter+1) % 10000 == 0:
            print (counter+1)
        betaEstimations.append(norm(linUCB.beta-linUCB.beta_estimation))
        Contexts_t = AllContexts
        counter += 1
        # Use our policy to choose next action
        arm, reward = linUCB.play(Contexts_t)
        # Save our action and observation
        arms.append(arm)
        rewards.append(reward)
        # Compute Best theoretical value
        bestArm, bestReward = linUCB.getBestReward(Contexts_t)
        # Save theoretical best action and observation
        bestArms.append(bestArm)
        bestRewards.append(bestReward)
        linUCB.computeEstimation()
    
    print Counter(arms)
    
    # Plots
    plot_regret(rewards, bestRewards, 'LinUCB', 'regret%d' % n_plot)
    plot_beta_estimation(betaEstimations, 'beta_estimation%d' % n_plot)
    plot_contexts_and_beta(AllContexts, theta, linUCB.beta_estimation, 'points%d' % n_plot)

    print linUCB.beta_estimation
    print linUCB.beta
    return


"""
Experiments with linear model without summation to 0
"""
beta = theta                        # no bias to sum to 0
alpha = 1.                          # exploration parameter
noise = np.random.normal(0,.4,T)    # noise
n_plot = 1
experiment(beta, alpha, noise, Contexts, n_plot)

"""
Experiments with linear model with summation to 0 (bias)
"""
bias = max(map(lambda x: np.dot(x,theta), Contexts))
beta = np.append(-bias, theta)      # no bias to sum to 0
AllContexts = map(lambda x: np.append(1,x), Contexts)
alpha = 1.                          # exploration parameter
noise = np.random.normal(0,.4,T)    # noise
n_plot = 2
experiment(beta, alpha, noise, AllContexts, n_plot)
