import numpy as np
import operator
import itertools
from modelConfig import params

class IntraOptionQLearning:
	def __init__(self, n_agents, discount, lr, terminations, weights):

		# param terminations: terminations is a list of termination objects over all the options
		# So, it's a vector of dimension (n_options, 1) i.e. 5 x 1 for us
		self.n_agents = n_agents
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
		self.last_joint_state = joint_state
		self.last_joint_option = joint_option
		self.last_value = self.getQvalue(joint_state, joint_option)

	def getQvalue(self, joint_state, joint_option=None, joint_option_in_use=None):
		# returns Q_mu(s,o)
		# joint_option = None converts this function to a value function that returns V_mu(s)
		if joint_option is None:
			# this returns the maximum value over all possible joint states
			
			# find all possible combination of joint options
			all_joint_options = list(self.weights[joint_state].keys())
			
			if not joint_option_in_use is None:
				all_joint_options.remove(joint_option_in_use)
			
			# calculate values for each of these joint options. One can call getQvalue here recursively
			all_Q = {option : self.getQvalue(joint_state, option) for option in all_joint_options}
			
			# return the maximum value and corresponding joint state
			max_idx, max_value = max(all_Q.items(), key=operator.itemgetter(1))
			
			return max_value
		
		return self.weights[joint_state][joint_option]
	
	def terminationProbOfAtLeastOneAgent(self, joint_state, joint_option):
		# calculates termination probability of at least one agent
		prod = 1.0
		for idx in range(self.n_agents):
			# prod *= 1 - self.terminations[self.agents[idx].option].pmf(joint_state[idx])
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
			beta = self.terminationProbOfAtLeastOneAgent(joint_state, joint_option) # 1-beta is the probability that
			# none of the agents terminate. Hence, the current option continues.
			update_target += self.discount*((1. - beta)*current_Q +
											beta * self.getQvalue(joint_state, None, joint_option))
			
			self.last_value = current_Q
			self.last_joint_option = joint_option
			self.last_joint_state = joint_state

		# Dense gradient update step
		tderror = update_target - self.last_value
		self.weights[self.last_joint_state] [self.last_joint_option] += self.lr*tderror

		return update_target


class IntraOptionActionQLearning:
	def __init__(self, n_agents, discount, lr, terminations, weights, qbigomega):
		self.n_agents = n_agents
		self.discount = discount
		self.lr = lr
		self.terminations = terminations #terminations is a list
		self.weights = weights
		self.qbigomega = qbigomega
		
	def start(self, joint_state, joint_option):
		'''
		:param joint_state: tuple of state encodings. Ranges from (0, 0, 0) to (103, 103, 103)
		:param joint_option:
		:return:
		'''
		self.last_joint_state = joint_state
		self.last_joint_option = joint_option
		self.last_value = self.getQvalue(joint_state, joint_option)

	def getQvalue(self, joint_state, joint_option, action):	#TODO: fix this
		return self.weights[joint_state][joint_option][action]
	
	def terminationProbOfAtLeastOneAgent(self, joint_state, joint_option):
		# calculates termination probability of at least one agent
		prod = 1.0
		for idx in range(self.n_agents):
			# prod *= 1 - self.terminations[self.agents[idx].option].pmf(joint_state[idx])
			prod *= 1 - self.terminations[joint_option[idx]].pmf(joint_state[idx])
		
		return 1.0 - prod

	def update(self, phi, joint_option, joint_action, reward, done):	#TODO: fix this
		# One-step update target
		update_target = reward
		if not done:
			current_values = self.qbigomega.value(phi)
			termination = self.terminations[self.last_options].pmf(Phi)
			one_or_more_termination_prob = self.one_or_more_terminate_prob(self.n_agents, self.terminations)
			update_target += self.discount*((1.-one_or_more_termination_prob)*current_values[self.last_jointOption] + one_or_more_termination_prob*np.max(current_values))

		# Update values upon arrival if desired
		tderror = update_target - self.value(self.last_Phi, self.last_jointOption, self.last_jointAction)
		self.weights[self.last_Phi, self.last_jointOption, self.last_jointAction] += self.lr*tderror

		self.last_Phi = phi
		self.last_jointOption = joint_option
		self.last_jointAction = joint_action