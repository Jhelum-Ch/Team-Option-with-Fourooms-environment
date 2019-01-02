import numpy as np
from scipy.misc import logsumexp
from modelConfig import params


class SoftmaxPolicy:
	def __init__(self, n_states, n_choices, temp=params['policy']['temperature']):
		'''
		
		:param n_states: number of encoded individual states
		:param n_choices: choices over which pmf spreads choice can be either number of primitive actions or options
		:param temp: lower temperature means uniform distribution, higher means delta
		'''
		self.rng = params['train']['seed']
		self.weights = np.zeros((n_states, n_choices))  # we assume that n_features and
		# n_actions for all agents are same
		self.temp = temp
	
	def value(self, phi, action=None):
		'''
		:param phi: state of the agent. Ranges from 1 to 103 (excluding walls)
		:param action:
		:return:
		'''
		if action is None:
			value = np.sum(self.weights[phi, :], axis=0)
		value = np.sum(self.weights[phi, action], axis=0)
		return value
	
	def pmf(self, phi):
		v = self.value(phi) / self.temp
		pmf = np.exp(v - logsumexp(v))
		return pmf
	
	def sample(self, phi):
		sample_action = int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))
		return sample_action


class EgreedyPolicy:
	def __init__(self, rng, n_features, n_actions, epsilon):
		self.rng = rng
		self.epsilon = epsilon
		self.weights = np.zeros((n_features, n_actions))
	
	def value(self, phi, action=None):
		if action is None:
			value = np.sum(self.weights[phi, :], axis=0)
		value = np.sum(self.weights[phi, action], axis=0)
		return value
	
	def sample(self, phi):
		if self.rng.uniform() < self.epsilon:
			sample_action = int(self.rng.randint(self.weights.shape[1]))
		sample_action = int(np.argmax(self.value(phi)))
		return sample_action


class FixedActionPolicies:
	def __init__(self, joint_action, n_actions):
		self.joint_action = joint_action
		self.probs = np.eye(n_actions)[joint_action]
	
	def sample(self, phi):
		return self.joint_action
	
	def pmf(self, phi):
		return self.probs