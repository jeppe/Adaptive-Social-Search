#!/bin/python

import numpy as np
import scipy.stats as stats


class LinearRewards:
    """
    Given a set of context, returns a reward at each time step
    """
    def __init__(self, beta, epsilon):
        """
        Parameters:
        ----------
        beta: numpy array (1*d)
            Unknown linear regressor we are trying to fit

        epsilon: noise
            Noise in the linear bandit assumtion: r = a^T x + eps
        """
        self.beta = beta # unknown vector we are trying to fit
        self.epsilon = epsilon

    def getReward(self, context):
        """
        context: numpy array (1*d)
        """
        return np.inner(self.beta, context) + self.epsilon.rvs()

class LinUCB:
    
    def __init__(self,d=2,alpha=0.1):
        # parameters of the algorithm
        self.d = d
        self.alpha = alpha
        # current estimations
        self.epoch = 0
        self.history = None
        self.betaEstimation_t = np.zeros(d)
        self.B_t = np.identity(self.d)
    
    def play(Contexts):
        """
        Contexts: numpy array c_t * d
        """
        self.betaEstimation_t = d

