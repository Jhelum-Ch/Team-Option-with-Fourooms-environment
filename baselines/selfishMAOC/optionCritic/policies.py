import numpy as np
from scipy.misc import logsumexp
from modelConfig import params, seed

class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            # return np.sum(self.weights[phi, :], axis=0)
            #print('weight',self.weights.shape[1])
            return self.weights[phi,:]
        else:    
            return np.sum(self.weights[phi, action], axis=0)        # TODO: Probably will have to remove sum here

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        #print('pmf',self.pmf(phi))
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))


# class SoftmaxOptionPolicy:
# 	def __init__(self, weights, temp=params['policy']['temperature']):
# 		'''
# 		:param temp: lower temperature means uniform distribution, higher means delta
# 		'''
# 		self.rng = seed
		
# 		self.weights = weights
# 		# weights is a dictionary keeping track of Q(s,o) values. This is the weight dictionary of IntraOptionQLearning
# 		self.temp = temp
	
# 	def pmf(self, joint_state):
# 		v = np.array(list(self.weights[joint_state].values())) / self.temp
# 		pmf = np.exp(v - logsumexp(v))
# 		return pmf
	
# 	def sample(self, joint_state):
# 		idx = int(self.rng.choice(len(self.weights[joint_state].keys()), p=self.pmf(joint_state)))
# 		joint_option = list(self.weights[joint_state].keys())[idx]
# 		return joint_option


	
# class SoftmaxActionPolicy:
# 	def __init__(self, n_states, n_choices, temp=params['policy']['temperature']):
# 		'''
# <<<<<<< HEAD
# =======

# >>>>>>> 987cde61614aea6df206a5a4447651ffc0113243
# 		:param n_states: number of encoded individual states
# 		:param n_choices: choices over which pmf spreads choice can be either number of primitive actions or options
# 		:param temp: lower temperature means uniform distribution, higher means delta
# 		'''
# 		self.rng = seed

# 		self.weights = np.zeros((n_states, n_choices))  # we assume that n_features and
# 		# n_actions for all agents are same
# 		self.temp = temp
	
# 	def value(self, phi, action=None):
# 		'''
# 		:param phi: state of the agent. Ranges from 1 to 103 (excluding walls)
# 		:param action:
# 		:return:
# 		'''
# 		if action is None:
# 			value = np.sum(self.weights[phi, :], axis=0)
# 		value = np.sum(self.weights[phi, action], axis=0)
# 		return value
	
# 	def pmf(self, phi):
# 		v = self.value(phi) / self.temp
# 		pmf = np.exp(v - logsumexp(v))
# 		return pmf
	
# 	def sample(self, phi):
# 		sample_action = int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))
# 		return sample_action
