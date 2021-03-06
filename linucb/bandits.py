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
        if np.linalg.norm( np.dot(self.B_inv, self.B) - np.eye(self.d) ) > 1e-10:
            print np.linalg.norm(np.dot(self.B_inv,self.B)-np.eye(self.d))
        return

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
        arm = 0
        for (i, x) in enumerate(Contexts):
            UCB = self.getUCB(x)
            if UCB > max_UCB:
                arm = i
                max_UCB = UCB
                best_context = x
        self.UCB = max_UCB
        return (arm, best_context)

    def getReward(self, context):
        """
        Generates a reward and update b
 
        Parameters:
        -----------
        context: numpy array (1*d)
        """
        self.reward = np.dot(self.beta, context) + self.epsilon[self.t-1]
        return self.reward

    def updateValues(self, context, reward):
        """
        Update the B matrix and the b numpy array
        """
        self.B += np.outer(context, context)
        self.b += reward * context

    def play(self, Contexts, init=True):
        """
        Parameters:
        -----------
        Contexts: numpy array c_t * d
        """
        self.t += 1
        if self.t <= len(Contexts) and init: # We play each arm once at the beginning
            arm = self.t - 1
            best_context = Contexts[self.t-1]
        else:
            arm, best_context = self.chooseArm(Contexts)
        reward = self.getReward(best_context)
        self.updateValues(best_context, reward)
        return arm, reward

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
            reward = np.dot(self.beta, x)
            if reward > bestReward:
                bestReward = reward
                bestArm = index
        return bestArm, bestReward


class LogisticUCB:
    """
    Given a set of context, returns a binary reward at each time step
    Modeled as a logistic regression.
    """
    def __init__(self, beta, alpha, epsilon):
        """
        Parameters:
        -----------
        beta: numpy array (1*d)
            Unknown linear regressor we are trying to fit

        epsilon: noise
            Noise in the linear bandit assumtion: r = a^T x + eps
        """
        # Parameters of generator
        self.beta = beta # unknown vector we are trying to fit
        self.d = len(self.beta)
        self.epsilon = epsilon # array of random samples of ]0,1[-uniform distribution
        self.alpha = alpha # exploration parameter (in front of confidence boundary)
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
        if np.linalg.norm( np.dot(self.B_inv, self.B) - np.eye(self.d) ) > 1e-10:
            print np.linalg.norm(np.dot(self.B_inv,self.B)-np.eye(self.d))
        return

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
        UCB = expit(np.dot(self.beta_estimation, context)) + \
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
        arm = 0
        for (i, x) in enumerate(Contexts):
            UCB = self.getUCB(x)
            if UCB > max_UCB:
                arm = i
                max_UCB = UCB
                best_context = x
        self.UCB = max_UCB
        return (arm, best_context)

    def getReward(self, context):
        self.reward = expit(np.dot(self.beta, context))
        if self.epsilon[self.t-1] < self.reward:
            self.reward = 1
        else:
            self.reward = 0
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
        if self.t <= len(Contexts):
            arm = self.t - 1
            best_context = Contexts[self.t-1]
        else:
            arm, best_context = self.chooseArm(Contexts)
        reward = self.getReward(best_context)
        self.updateValues(best_context, reward)
        return arm, reward

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


"""
TODO
"""
#class LogisticThomsonSampling
