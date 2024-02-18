import gym
import argparse
import numpy as np
import random

from scipy.special import expit
from scipy.special import logsumexp


class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class SoftmaxPolicy:
    def __init__(self, nfeatures, nactions, temp=1.):
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        exp_v = np.exp(v - logsumexp(v))
        normalized_pmf = exp_v / np.sum(exp_v)
        return normalized_pmf
    
    def all_pmfs(self,):
        all_pmfs=[]
        for phi in range(len(self.weights)):
            v = self.value([phi])/self.temp
            all_pmfs.append(np.exp(v - logsumexp(v)))
        return np.array(all_pmfs)

    def sample(self, phi, mask = None):
        prob = self.pmf(phi)
        prob = prob[0]
        if mask is not None:
            prob -= mask*1e5
            prob = np.exp(prob - np.max(prob))
            prob = prob / prob.sum(axis=0)
        # return int(np.random.choice(self.weights.shape[1], p=self.pmf(phi)))
        else:
            prob = prob[0]
        return int(np.random.choice(np.arange(self.weights.shape[1]), p = prob))

class SigmoidTermination:
    def __init__(self, nfeatures):
        self.weights = np.zeros((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(np.random.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi

class SigmoidInterestFunction:
    def __init__(self,):
        self.room0 = list(range(5)) + list(range(10, 15)) + list(range(20, 26)) + list(range(31, 36)) + list(range(41, 46)) + [51]
        self.room1 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57))
        self.room2 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) + [62]
        self.room3 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 89)) + list(range(94, 99))
        self.rooms = [self.room0, self.room1, self.room2, self.room3]

    def get(self, phi, option):
        interest = float(phi in self.rooms[option])
        if interest == 0.:
            interest=0.1
        return interest

    def getall(self, phi):
        interest= np.ones((4)) * 0.1
        for o in range(4):
            if phi in self.rooms[o]:
                interest[o] = 1.
        return interest

class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights, meta_policy, noptions, initiation_policy=None):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.meta_policy = meta_policy
        self.noptions=noptions
        self.initiation_policy=initiation_policy

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi)
        

    def value(self, phi, option=None):
        if option is None:
            value = np.sum(self.weights[phi, :], axis=0)
        else:
            value = np.sum(self.weights[phi, option], axis=0)
        return value[0]

    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - self.meta_policy.pmf(phi).dot(values)
        if option is None:
            return advantages
        return advantages[option]

    def uponarrival(self, next_phi, option):
        qvalues = self.value(next_phi,option)
        all_values = self.value(next_phi)
        if self.initiation_policy is  None:
            values = self.meta_policy.pmf(next_phi).dot(all_values)
            beta = self.terminations[option].pmf(next_phi)
        else:
            values = np.array(self.initiation_policy.pmf(next_phi)).dot(all_values)
            beta = self.terminations.pmf(next_phi,option)
        u_so = qvalues * (1-beta) + values * beta
        return u_so


    def update(self, next_phi, next_option, reward, done, multiplier):

        # One-step update target
        update_target = reward
        current_values = self.value(next_phi)
        if not done:
            update_target += self.discount * current_values[next_option]

        
        # Dense gradient update step
        tderror = update_target - self.last_value[self.last_option]


        for o in range (self.noptions):
            self.weights[self.last_phi, o] += self.lr*multiplier[o] * \
            (reward + self.discount * float(not done) * self.uponarrival(next_phi,o) - self.last_value[o] )
            

        self.last_value = self.value(next_phi)
        self.last_option = next_option
        self.last_phi = next_phi

class TerminationGradient:
    def __init__(self, terminations, critic, lr,noptions):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr
        self.noptions=noptions

    def update(self, phi, option, multiplier):
        for o in range(self.noptions):
            magnitude, direction = self.terminations[o].grad(phi)
            self.terminations[o].weights[direction] -= \
                    self.lr*multiplier[o]*magnitude*(self.critic.advantage(phi, o))             

class IntraOptionGradient:
    def __init__(self, option_policies, lr, discount, critic,noptions):
        self.lr = lr
        self.option_policies = option_policies
        self.discount = discount
        self.critic= critic
        self.noptions=noptions

    def update(self, phi, option, action, reward, done, next_phi, next_option, critic, multiplier, prob_cur_option=1.):

        for o in range(self.noptions):

            adv =(reward + self.discount * float(not done) * self.critic.uponarrival(next_phi,o) - self.critic.value(phi,o))
            actions_pmf = self.option_policies[o].pmf(phi)
            mult_o = multiplier[o]
            self.option_policies[o].weights[phi, :] -= self.lr* mult_o *prob_cur_option[o]*adv*actions_pmf
            self.option_policies[o].weights[phi, action] += self.lr* mult_o*prob_cur_option[o]*adv

class InitiationSetSoftmaxPolicy:
    def __init__(self, noptions, initiationset, poveroptions):
        self.noptions = noptions
        self.initiationset = initiationset
        self.poveroptions = poveroptions

    def pmf(self, phi, option=None):
        list1 = [self.initiationset.get(phi,opt) for opt in range(self.noptions)]   
        list2 = self.poveroptions.pmf(phi)
        normalizer = sum([x * y for x, y in zip(list1, list2)])
        pmf = [float(list1[i] * list2[i])/normalizer for i in range(self.noptions)]
        return pmf

    def sample(self, phi):
        # import pdb;pdb.set_trace()
        return int(np.random.choice(self.noptions, p=self.pmf(phi)))

    def all_pmfs(self,):
        all_pmfs=[]
        for phi in range(len(self.poveroptions.weights)):
            pmf = self.pmf(np.array([phi]))
            all_pmfs.append(pmf)
        return np.array(all_pmfs)
