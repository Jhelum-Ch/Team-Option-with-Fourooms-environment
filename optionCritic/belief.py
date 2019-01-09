import numpy as np
import scipy.stats
import random
from modelConfig import params


class Belief:
	def __init__(self, env, sample_count=1000):
		super(Belief, self).__init__()
		self.env = env.deepcopy()
		self.sample_count = 1000
		self.curr_joint_state = env.currstate
		self.states_list = env.states_list
		self.p_val = np.full(len(env.states_list))
		self.p_val[(self.curr_joint_state)] = 1.0
		
	def rejectionSampling(self):
		
		# determine neighborhood of each agent
		# neighborhood =  np.array((params['env']['n_agents'], params['agent']['n_actions']))
		# for agent in range(params['env']['n_agents']):
		# 	for action in range(params['agent']['n_actions']):
		# 		self.env.currstate = self.curr_joint_state
		# 		self.env.step()
		# 		neighborhood[agent, action] = self.env.currstate[agent]
				
		# each agent rejects a next state based on it's own neighborhood
		consistent = False
		sample_count = 0
		while consistent is False and sample_count <= self.sample_count:
			sampled_joint_state = self.sampleJointState()
			for agent in range(params['env']['n_agents']):
				# TODO : check for consistency
			
				if not consistent:
					consistent = False
					break
			sample_count += 1
			
		return sampled_joint_state
				
	
	def sampleJointState(self):
		# this is executed by the co-ordinator
		sampled_state_idx = int(np.random.choice(len(self.states_list), size=1, p=self.p_val))
		return self.states_list[sampled_state_idx]
		#return np.random.multinomial(1, self.p_val)


# class Belief:
# 	#
# 	'''
# 	Implemented as in https://en.wikipedia.org/wiki/Conjugate_prior
# 	Help : https://stats.stackexchange.com/questions/312802/how-to-set-the-priors-for-bayesian-estimation-of-multivariate-normal-distributio
# 	'''
#
# 	def __init__(self, env, sample_count=1000):
#
# 		# super(multivariateNormalBelief, self).__init__()
# 		# D : dimension
# 		self.D = env.n_agents
# 		self.states_list = env.states_list
#
# 		'''
# 		Priors
# 		Query: Is it a good idea to set the dimension = number of agents?
# 		'''
# 		self.mu0 = np.full(env.n_agents, 1.0/env.n_agents) #np.zeros(self.D)
# 		self.cov0 = np.eye(self.D)
# 		np.fill_diagonal(self.cov0, 1.0/env.n_agents)
#
# 		# k_0 (conflict of definition with the wikipedia page)
# 		self.k0 = 0  # 0.1
# 		self.v0 = self.D + 2  # self.D + 1.5
# 		assert isinstance(self.k0, int) and isinstance(self.v0, int) == True, 'k0 and v0 must be integers'
#
# 		self.psi = (self.v0 - self.D - 1) * np.identity(self.D)
#
# 		# Number of samples
# 		self.N = sample_count
#
# 		self.num_itr = 100
#
# 		self.mean_itr = np.random.uniform(0, 1, self.D)
# 		# self.mean_itr = random.sample(self.states_list, k=self.D)
# 		self.cov_itr = scipy.stats.invwishart.rvs(self.v0, self.psi)
#
# 	def sample(self):
# 		'''
# 		purpose : samples observation from the current belief distribution
# 		returns : data matrix of dimenson (number of agents x sample_Count)
# 		'''
# 		samples = np.random.multivariate_normal(mean=self.mean_itr, cov=self.cov_itr, size=self.N)
# 		return samples
#
# 	def sampleJointState(self):
# 		'''
# 		purpose : sample a unique state from the current belief distribution
# 		returns : joint state tuple
# 		'''
# 		sample = np.random.multivariate_normal(mean=self.mean_itr, cov=self.cov_itr, size=1)
#
# 		state_list = []
# 		for i in range(sample.size):
# 			state_list.append(sample[0][i])
#
# 		return tuple(state_list)
#
# 	def updateBeliefParameters(self, samples):
# 		'''
# 		uses Normal Inverse Wishart for posterior update of the parameters of the prior distribution
# 		'''
# 		x_bar = np.mean(samples, axis=0)  # sample mean
# 		print(x_bar.size)
# 		sample_cov = np.cov(samples)
#
# 		# Gibb's sampling
# 		k = self.k0 + self.N
# 		v = self.v0 + self.N
#
# 		for _ in range(self.num_itr):
# 			# Update mean
# 			mean_tmp = (self.k0 * self.mu0 + self.N * x_bar) / (self.k0 + self.N)
# 			print(mean_tmp)
# 			self.mean_itr = np.random.multivariate_normal(mean_tmp, self.cov_itr / k, 1)
#
# 			# Update cov
# 			sample_demean = samples - self.mean_itr
# 			C = np.dot((samples - self.mean_itr).T, (samples - self.mean_itr))
# 			scale_tmp = self.psi + C + (self.k0 * self.N) / (self.k0 + self.N) * np.dot((x_bar - self.mu0).T,
# 																						(x_bar - self.mu0))
# 			self.cov_itr = scipy.stats.invwishart.rvs(v, scale_tmp)

	
	
