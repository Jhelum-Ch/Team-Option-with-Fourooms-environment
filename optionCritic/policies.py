import numpy as np
from scipy.misc import logsumexp
from modelConfig import params


class SoftmaxPolicy:
	def __init__(self, weights, temp=params['policy']['temperature']):
		'''
		:param temp: lower temperature means uniform distribution, higher means delta
		'''
		self.rng = params['train']['seed']
		
		self.weights = weights
		# weights is a dictionary keeping track of Q(s,o) values. This is the weight dictionary of IntraOption Q
		# learning
		self.temp = temp
	
	def getQvalue(self, agent_state, action=None):
		# 1. select all rows where agent_state appears
		agent_state_keys = [s for s in self.weights.keys() if agent_state in s]
		# print(agent_state_keys)
		# print(len(agent_state_keys))
		
		# 2. for each option:
		# 		for each row:
		# 			calculate the sum of Q values where option appears in joint_options for that row
		
		Q = np.zeros((params['agent']['n_options']))
		
		for option in range(params['agent']['n_options']):
			for joint_state in agent_state_keys:
				# print(joint_state)
				option_keys = [o for o in self.weights[joint_state].keys() if option in o]
				# print(option_keys)
				for o in option_keys:
					Q[option] += self.weights[joint_state][o]	#TODO : need to average instead of sum
		# print(Q)
		
		# TODO : mask Q for unavailable options
	
		return Q
		
	
	def pmf(self, agent_state):
		v = self.getQvalue(agent_state) / self.temp
		pmf = np.exp(v - logsumexp(v))
		return pmf
	
	def sample(self, joint_state):
		joint_option = []
		for agent_state in joint_state:
			sample_option = int(self.rng.choice(range(params['agent']['n_options']), p=self.pmf(agent_state)))
			joint_option.append(sample_option)
			#TODO : make the sampled option unavailable
		return tuple(joint_option)
