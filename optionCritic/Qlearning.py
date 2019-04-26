import numpy as np
import operator
import itertools
from modelConfig import params

class IntraOptionQLearning:
	def __init__(self, discount, lr, terminations, weights):

		# param terminations: terminations is a list of termination objects over all the options
		# So, it's a vector of dimension (n_options, 1) i.e. 5 x 1 for us
		# self.n_agents = n_agents
		self.discount = discount
		self.lr = lr
		self.terminations = terminations
		self.weights = weights	#let's assume weights are dictionary

	def start(self, joint_state, joint_option):
		'''
		:param joint_state: tuple of state encodings. Ranges from (0, 0, 0) to (103, 103, 103)
		:param joint_option:
		:return:
		'''
		self.last_joint_state = tuple(np.sort(joint_state))
		self.last_joint_option = tuple(np.sort(joint_option))
		self.last_value = self.getQvalue(joint_state, joint_option)

	def getQvalue(self, joint_state, joint_option=None, joint_option_in_use=None):
		# returns Q_mu(s,o)
		# joint_option = None converts this function to a value function that returns V_mu(s)
		joint_state = tuple(np.sort(joint_state))
		if joint_option:
			joint_option = tuple(np.sort(joint_option))
		
		if joint_state not in self.weights.keys():
			self.weights[joint_state] = {}
			if joint_option:
				self.weights[joint_state][joint_option] = 0.0
			
		if joint_option is None:
			# this returns the maximum value over all possible joint states
			
			# find all possible combination of joint options
			all_joint_options = list(self.weights[joint_state].keys())
			
			if joint_option_in_use:
				all_joint_options.remove(joint_option_in_use)
			
			# calculate values for each of these joint options. One can call getQvalue here recursively
			all_Q = {option : self.getQvalue(joint_state, option) for option in all_joint_options}
			
			if not all_Q:
				return 0	# accounts for advantage function when Q(s,o) has never been called before for any o
				# import  pdb; pdb.set_trace()
			
			# return the maximum value and corresponding joint state
			max_idx, max_value = max(all_Q.items(), key=operator.itemgetter(1))
			
			return max_value
<<<<<<< HEAD


		if joint_option not in self.weights[joint_state].keys():
				self.weights[joint_state][joint_option] = 0.0
		
		
=======
		
		if joint_option not in self.weights[joint_state].keys():
				self.weights[joint_state][joint_option] = 0.0
		
>>>>>>> 48826ef82b860ce6142d604849e7dc2331368dee
		return self.weights[joint_state][joint_option]
	
	def terminationProbOfAtLeastOneAgent(self, joint_state, joint_option):
		# calculates termination probability of at least one agent
		prod = 1.0
		for idx in range(len(joint_state)):
			prod *= 1 - self.terminations[joint_option[idx]].pmf(joint_state[idx])
			
		return 1.0 - prod

	def getAdvantage(self, joint_state, joint_option=None):
		v = self.getQvalue(joint_state)
		if joint_option is None:
			return v #TODO: check if this should be -v
		q = self.getQvalue(joint_state, joint_option)
		return q - v

	def update(self, joint_state, joint_option, reward, done):
		# One-step update target
		update_target = reward	#delta
		current_Q = self.getQvalue(joint_state, joint_option)
		if not done:
			beta = self.terminationProbOfAtLeastOneAgent(joint_state, joint_option) # (1 - beta) is the probability
			# that none of the agents terminate. Hence, the current option continues.
			update_target += self.discount*((1. - beta)*current_Q + beta * self.getQvalue(joint_state, None,
																						  joint_option))
			
			self.last_value = current_Q
			self.last_joint_option = tuple(np.sort(joint_option))
			self.last_joint_state = tuple(np.sort(joint_state))

		# Dense gradient update step
		tderror = update_target - self.last_value
		self.weights[self.last_joint_state] [self.last_joint_option] += self.lr*tderror

		return update_target


class IntraOptionActionQLearning:
	def __init__(self, discount, lr, terminations, qbigomega):
		# self.n_agents = n_agents
		self.discount = discount
		self.lr = lr
		self.terminations = terminations #terminations is a list
		self.weights = {}
		self.qbigomega = qbigomega
		
	def start(self, joint_state, joint_option, joint_action):
		'''
		:param joint_state: tuple of state encodings. Ranges from (0, 0, 0) to (103, 103, 103)
		:param joint_option:
		:return:
		'''
		self.last_joint_state = tuple(np.sort(joint_state))
		self.last_joint_option = tuple(np.sort(joint_option))
		self.last_joint_action = joint_action

	def getQvalue(self, joint_state, joint_option, joint_action):
		joint_state = tuple(np.sort(joint_state))
		joint_option = tuple(np.sort(joint_option))
		
		if joint_state not in self.weights.keys():
			self.weights[joint_state] = {}
			self.weights[joint_state][joint_option] = {}
			self.weights[joint_state][joint_option][joint_action] = 0.0
			
		elif joint_option not in self.weights[joint_state].keys():
			self.weights[joint_state][joint_option] = {}
			self.weights[joint_state][joint_option][joint_action] = 0.0
		
		elif joint_action not in self.weights[joint_state][joint_option].keys():
			self.weights[joint_state][joint_option][joint_action] = 0.0
			
		return self.weights[joint_state][joint_option][joint_action]
	
	def terminationProbOfAtLeastOneAgent(self, joint_state, joint_option):
		# calculates termination probability of at least one agent
		prod = 1.0
		for idx in range(len(joint_state)):
			prod *= 1 - self.terminations[joint_option[idx]].pmf(joint_state[idx])

		return 1.0 - prod

	def update(self, joint_state, joint_option, joint_action, reward, done):
		
		# One-step update target
		update_target = reward
		if not done:
			current_Q_option = self.qbigomega.getQvalue(joint_state, self.last_joint_option)
			beta = self.terminationProbOfAtLeastOneAgent(joint_state, self.last_joint_option)
			#termination = self.terminations[self.last_options].pmf(Phi)
			update_target += self.discount * ((1.- beta) * current_Q_option + beta * self.qbigomega.getQvalue(
				joint_state, None, self.last_joint_option))

		tderror = update_target - self.getQvalue(self.last_joint_state, self.last_joint_option, self.last_joint_action)
		self.weights[tuple(np.sort(self.last_joint_state))][tuple(np.sort(self.last_joint_option))][self.last_joint_action] += self.lr*tderror
		
		self.last_joint_state = joint_state
		self.last_joint_option = joint_option
		self.last_joint_action = joint_action
		
# class AgentQLearning:
# 	def __init__(self, discount, lr, weights):
# 		self.discount = discount
# 		self.lr = lr
# 		# n_options = params['agent']['n_options']
# 		# n_actions = params['agent']['n_actions']
# 		self.weights = weights	#oder of weights is option x state x action
#
# 	def start(self, joint_state, joint_option, joint_action):
# 		self.last_joint_state = joint_state
# 		self.last_joint_option = joint_option
# 		self.last_joint_action = joint_action
#
# 	def value(self, state, option, action):
# 		return self.weights[option, state, action]
#
# 	def update(self, joint_state, joint_option, joint_action, reward, done):
# 		update_target = [reward] * len(joint_state)
# 		if not done:
# 			for idx, state, option in zip(range(len(joint_state)), joint_state, joint_option):
# 				update_target[idx] += self.discount * np.max(self.weights[option, state, :])
#
# 		for idx, last_state, last_option, last_action in zip(range(len(self.last_joint_state)), self.last_joint_state,
# 															 self.last_joint_option, self.last_joint_action):
# 			tderror = update_target[idx] - self.value(last_state, last_option, last_action)
# 			self.weights[last_option, last_state, last_action] += self.lr * tderror
#
# 		self.last_state = joint_state
# 		self.last_option = joint_option
# 		self.last_action = joint_action

class AgentQLearning:
	def __init__(self, discount, lr, options):
		self.discount = discount
		self.lr = lr
		# n_options = params['agent']['n_options']
		# n_actions = params['agent']['n_actions']
		self.options = options  # oder of weights is option x state x action

	def start(self, joint_state, joint_option, joint_action):
		self.last_joint_state = joint_state
		self.last_joint_option = joint_option
		self.last_joint_action = joint_action

	def value(self, state, option, action):
		return self.options[option].policy.weights[state, action]

	def update(self, joint_state, joint_option, joint_action, reward, done):
		update_target = [reward] * len(joint_state)
		if not done:
			for idx, state, option in zip(range(len(joint_state)), joint_state, joint_option):
				action_pmfs = self.options[option].policy.weights[state, :]
				update_target[idx] += self.discount * np.max(action_pmfs)

		for idx, last_state, last_option, last_action in zip(range(len(self.last_joint_state)), self.last_joint_state,
															 self.last_joint_option, self.last_joint_action):
			tderror = update_target[idx] - self.value(last_state, last_option, last_action)
			self.options[last_option].policy.weights[last_state, last_action] += self.lr * tderror

		self.last_state = joint_state
		self.last_option = joint_option
		self.last_action = joint_action
		
		
		