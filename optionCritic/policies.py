import numpy as np
from scipy.misc import logsumexp
from modelConfig import params


class SoftmaxOptionPolicy:
	def __init__(self, weights, temp=params['policy']['temperature']):
		'''
		:param temp: lower temperature means uniform distribution, higher means delta
		'''
		self.rng = params['train']['seed']
		
		self.weights = weights
		# weights is a dictionary keeping track of Q(s,o) values. This is the weight dictionary of IntraOptionQLearning
		self.temp = temp
	
	def pmf(self, joint_state):
		v = np.array(list(self.weights[joint_state].values())) / self.temp
		pmf = np.exp(v - logsumexp(v))
		return pmf
	
	def sample(self, joint_state):
		idx = int(self.rng.choice(len(self.weights[joint_state].keys()), p=self.pmf(joint_state)))
		joint_option = list(self.weights[joint_state].keys())[idx]
		# TODO : make the sampled option unavailable
		return joint_option

# class SoftmaxOptionPolicy:
# 	def __init__(self, weights, temp=params['policy']['temperature']):
# 		'''
# 		:param temp: lower temperature means uniform distribution, higher means delta
# 		'''
# 		self.rng = params['train']['seed']
#
# 		self.weights = weights
# 		# weights is a dictionary keeping track of Q(s,o) values. This is the weight dictionary of IntraOption Q
# 		# learning
# 		self.temp = temp
#
# 	def getQvalue(self, agent_state, action=None):
# 		# 1. select all rows where agent_state appears
# 		agent_state_keys = [s for s in self.weights.keys() if agent_state in s]
# 		# print(agent_state_keys)
# 		# print(len(agent_state_keys))
#
# 		# 2. for each option:
# 		# 		for each row:
# 		# 			calculate the sum of Q values where option appears in joint_options for that row
#
# 		Q = np.zeros((params['agent']['n_options']))
#
# 		for option in range(params['agent']['n_options']):
# 			for joint_state in agent_state_keys:
# 				# print(joint_state)
# 				option_keys = [o for o in self.weights[joint_state].keys() if option in o]
# 				# print(option_keys)
# 				for o in option_keys:
# 					Q[option] += self.weights[joint_state][o]	#TODO : need to average instead of sum
# 		# print(Q)
#
# 		# TODO : mask Q for unavailable options
#
# 		return Q
#
#
# 	def pmf(self, agent_state):
# 		v = self.getQvalue(agent_state) / self.temp
# 		pmf = np.exp(v - logsumexp(v))
# 		return pmf
#
# 	def sample(self, joint_state):
# 		#TODO: fix for option selection for individual agent
# 		joint_option = []
# 		for agent_state in joint_state:
# 			sample_option = int(self.rng.choice(range(params['agent']['n_options']), p=self.pmf(agent_state)))
# 			joint_option.append(sample_option)
# 			#TODO : make the sampled option unavailable
# 		return tuple(joint_option)
	
	
	
class SoftmaxActionPolicy:
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
