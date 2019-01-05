import numpy as np
from itertools import combinations
import operator

class IntraOptionQLearning:
	def __init__(self, n_agents, discount, lr, terminations, weights):

		# param terminations: terminations is a list of termination objects over all the options
		# So, it's a vector of dimension (n_options, 1) i.e. 5 x 1 for us
		# self.agents = agents	# holds agents objects
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
		self.last_value = self.getQvalue(joint_state, joint_option)		# TODO: need a scalar

	def getQvalue(self, joint_state, joint_option=None):
		if joint_option is None:
			return self.weights[joint_state]
		return self.weights[joint_state] [joint_option]
	
	def terminationProbOfAtLeastOneAgent(self, joint_state, joint_option):
		# calculates termination probability of at least one agent
		prod = 1.0
		for idx in range(self.n_agents):
			# prod *= 1 - self.terminations[self.agents[idx].option].pmf(joint_state[idx])
			prod *= 1 - self.terminations[joint_option[idx]].pmf(joint_state[idx])
			
		return 1.0 - prod

	def getAdvantage(self, joint_state, joint_option=None):
		values = self.getQvalue(joint_state)
		max_idx, max_value = max(values.items(), key=operator.itemgetter(1))
		advantages = {key:val - max_value for key, val in values.items()}
		if joint_option is None:
			return advantages
		return advantages[joint_option]

	def update(self, joint_state, joint_option, reward, done):
		# One-step update target
		update_target = reward
		if not done:
			current_values = self.getQvalue(joint_state)
			max_current_value = max(current_values.items(), key=operator.itemgetter(1))[1]
			#termination = self.terminations[self.last_jointOption].pmf(phi)

			#modify this according to current writeup
			one_or_more_termination_prob = self.terminationProbOfAtLeastOneAgent(joint_state, joint_option)
			update_target += self.discount*((1.-one_or_more_termination_prob)*current_values[self.last_joint_option] +
											one_or_more_termination_prob*max_current_value)
			
			self.last_value = current_values[joint_option]
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

	def value(self, phi, joint_option, joint_action): #Phi is a matrix, joint_option and joint_action are lists
		out[i] = np.sum(self.weights[phi, joint_option[i], joint_action[i]], axis=0)

		return np.sum(out)

	def one_or_more_terminate_prob(self, n_agents, terminations):
		superset = [list(combinations(range(n_agents)))]
		superset.remove(set())

		sumtotal = 0.0
		for item in superset:
			product = 1.0
			for i in item:
				product *= terminations[item[i]]

			sumtotal += product

		return sumtotal

	def start(self, phi, joint_option, joint_action): #Phi is a matrix, joint_option and joint_action are lists
		self.last_Phi = phi
		self.last_jointOption = joint_option
		self.last_jointAction = joint_action

	def update(self, phi, joint_option, joint_action, reward, done):
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