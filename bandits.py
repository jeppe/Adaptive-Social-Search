#!/bin/python

import numpy as np
import scipy.stats as stats
from scipy.special import expit
import random
import sys

MIN_INT = -sys.maxint - 1

class LinearUCB:
    """
    Given a set of context, returns a reward at each time step
    """
    def __init__(self, beta, alpha, epsilon):
        """
        Parameters:
        ----------
        beta: numpy array (1*d)
            Unknown linear regressor we are trying to fit

        epsilon: noise
            Noise in the linear bandit assumtion: r = a^T x + eps
        """
        # Parameters of generator
        self.beta = beta # unknown vector we are trying to fit
        self.d = len(self.beta)
        self.epsilon = epsilon
        self.alpha = alpha
        self.t = 0
        # Estimations
        self.beta_estimation = None
        self.B = np.eye(self.d)
        self.B_inv = None
        self.b = np.zeros(self.d)
        self.reward = None
        self.UCB = None

    def computeEstimation(self):
        self.B_inv = np.linalg.inv(self.B)
        self.beta_estimation = np.dot(self.B_inv, self.b)

    def getUCB(self, context):
        """
        Compute the upper confidence bound of the linear arm for this context

        Parameters:
        -----------
        context: numpy array (1 * d)
        B_inv: numpy array (d * d)
        t: int
        """
        bilinear = np.dot(context, np.dot(self.B_inv, context))
        UCB = np.dot(self.beta_estimation, context) + \
                self.alpha * np.sqrt(bilinear * np.log(self.t+2))
        return UCB

    def chooseArm(self, Contexts):
        """
        Chooses the next arm to play among the vectors in Contexts

        Parameters:
        -----------
        Contexts: list of context vectors
        """
        max_UCB = MIN_INT
        best_context = None
        index = 0
        for (i, x) in enumerate(Contexts):
            UCB = self.getUCB(x)
            if UCB > max_UCB:
                index = i
                max_UCB = UCB
                best_context = x
        self.UCB = max_UCB
        return (index, best_context)

    def getReward(self, context):
        """
        Generates a reward and update b
 
        Parameters:
        -----------
        context: numpy array (1*d)
        """
        #self.reward = expit(np.dot(self.beta, context)) # + self.epsilon.rvs()
        self.reward = np.dot(self.beta, context) + self.epsilon.rvs()
        #if random.random() < self.reward:
        #    self.reward = 1
        #else:
        #    self.reward = 0
        self.b += self.reward * context
        return self.reward

    def updateValues(self, context, reward):
        """
        Update the B matrix and the b numpy array
        """
        self.B += np.outer(context, context)
        self.b += reward * context

    def play(self, Contexts):
        """
        Parameters:
        -----------
        Contexts: numpy array c_t * d
        """
        self.t += 1
        #self.computeEstimation()
        index, best_context = self.chooseArm(Contexts)
        reward = self.getReward(best_context)
        self.updateValues(best_context, reward)
        return index, reward

    def getBestReward(self, Contexts):
        """
        Compute the mean reward of the best arm

        Parameters:
        -----------
        Contexts: numpy array c_t * d
        """
        bestReward = MIN_INT
        bestArm = None
        for index, x in enumerate(Contexts):
            reward = expit(np.dot(self.beta, x))
            if reward > bestReward:
                bestReward = reward
                bestArm = index
        return bestArm, bestReward
