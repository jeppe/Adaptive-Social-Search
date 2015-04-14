#!/usr/bin/env python
#-*-coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_regret(rewards, bestRewards, label, filename):
    sumRewards = np.cumsum(rewards)
    sumBestRewards = np.cumsum(bestRewards)
    regret = (sumBestRewards - sumRewards)
    fig = plt.figure(figsize=(7, 6))
    plt.plot(np.arange(1,len(regret)+1), regret, label=label)
    plt.legend()
    plt.savefig('./img/'+filename+'.png')
    plt.close(fig)
    return

def plot_beta_estimation(betaEstimations, filename):
    fig = plt.figure(figsize=(7, 6))
    plt.axis([0, len(betaEstimations), 0, 1.])
    plt.plot(np.arange(1,len(betaEstimations)+1),betaEstimations,label='loss distance distance')
    plt.legend()
    plt.savefig('./img/'+filename+'.png')
    plt.close(fig)
    return

def plot_contexts_and_beta(AllContexts, theta, beta_estimation, filename):
    fig = plt.gcf()
    plt.gca().set_xlim((0.,1.2))
    plt.gca().set_ylim((0.,1.2))
    plt.gca().plot(np.array(map(lambda x: x[0], AllContexts)),  # plot context vectors
                   np.array(map(lambda x: x[1], AllContexts)),
                   'o',color='black')
    plt.gca().plot(theta[0], theta[1],'o',color='blue')         # plot theta vector (hidden vector)
    normalisation = LA.norm(np.array([beta_estimation[0], beta_estimation[1]]))
    plt.gca().plot(beta_estimation[0] / normalisation,          # plot beta estimation
                   beta_estimation[1] / normalisation,
                   'o',color='red')
    fig.gca().add_artist(plt.Circle((0,0),1.,color='b',fill=False))
    for i, x in enumerate(AllContexts):
        fig.gca().annotate('%d' % i, xy=(x[0],x[1]), xytext=(x[0], x[1]),
                           arrowprops=dict(facecolor='black', shrink=0.05),
                          )
    fig.savefig('img/'+filename+'.png')
