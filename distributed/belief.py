import collections
import numpy as np
import scipy.stats as stats
import random
from modelConfig import params
import copy
from fourroomsEnv import FourroomsMA


class MultinomialDirichletBelief:
	def __init__(self, env, joint_observation, sample_count=10):
		super(MultinomialDirichletBelief, self).__init__()
		self.env = copy.deepcopy(env)
		# self.env = env.deepcopy()
		# self.joint_observation = joint_observation
		self.sample_count = sample_count  # sample_count is for rejection sampling
		self.curr_joint_state = self.env.currstate
		self.states_list = self.env.states_list
		
		self.alpha = 0.001 * np.ones(len(self.states_list))
		# randomly pick an idx of alpha and make the peak of delta at it
		idx = int(np.random.choice(range(len(self.states_list)), 1))  # uniformly choose a joint-state index
		self.alpha[int(idx)] += 1.  # make a delta at the chosen state
		
		self.num_type = len(self.alpha)  # number of types
	
	def posteriorPMF(self):
		counts = collections.Counter(self.states_list)
		counts_vec = [counts.get(i, 0) for i in range(self.num_type)]
		if [self.joint_observation[i] is None for i in range(len(self.joint_observation))]:
			a = stats.dirichlet.rvs(self.alpha, size=1, random_state=1)
			return a[0]  # return 1 random sample
		
		elif [not self.joint_observation[i] is None for i in range(len(self.joint_observation))]:
			# if all agents broadcast, set prosterior to delta
			self.alpha = 0.001 * np.ones(len(self.states_list))
			idx = np.random.choice(range(len(self.states_list)), 1)
			self.alpha[int(idx)] += 1.
			return self.alpha
		else:
			list_of_not_none = [i for i in range(len(self.joint_observation)) if
								self.joint_observation[i] != None]  # find indices of not None in y
			
			''' find list of keys in counts which has non-None components of y and all possible values
			for None component of y'''
			
			b = []
			for item in list(counts.keys()):
				
				flag = 0
				for i in list_of_not_none:
					if item[i] == self.joint_observation[i]:
						flag += 1
				if flag == len(list_of_not_none):
					b.append(item)
			
			for item in b:
				counts[item] += 1
			
			counts_vec = [counts.get(i, 0) for i in range(self.num_type)]
			a = stats.dirichlet.rvs(np.add(self.alpha, counts_vec), size=1, random_state=1)
			return a[0]  # return 1 random sample
	
	def sampleJointState(self, joint_observation):  # sample one joint_state from posterior
		self.joint_observation = joint_observation
		sampled_state_idx = int(np.random.choice(range(len(self.states_list)), 1, p=self.posteriorPMF()))
		return self.states_list[sampled_state_idx]
	
	def rejectionSampling(self):
		
		# each agent rejects a sample from common-belief posterior based on its own true state
		true_joint_state = self.env.currstate
		consistent = False
		sample_count = 0
		rs = np.zeros(params['env']['n_agents'])
		while consistent is False and sample_count <= self.sample_count:
			sampled_joint_state = self.sampleJointState()
			for agent in range(params['env']['n_agents']):
				# rejection sampling
				rs[agent] = 1.0 * (true_joint_state[agent] == sampled_joint_state[
					agent])  # agent accepts if the sampledjoint-state has its true state
			if np.prod(rs) == 1.0:
				consistent = True
				if not consistent:
					consistent = False
					break
			sample_count += 1
		
		return sampled_joint_state
	
	def rejectionSamplingNeighbour(self):
		
		# determine neighborhood of each agent
		neighborhood = np.empty((params['env']['n_agents'], params['agent']['n_actions']))  # create an empty n-d-array
		for agent in range(params['env']['n_agents']):
			for action in range(params['agent']['n_actions']):
				self.env.currstate = self.curr_joint_state
				neighboring_state = self.env.neighbouringState(agent, action)
				neighborhood[agent, action] = neighboring_state
		
		# each agent rejects a sample from common-belief posterior based on its own neighborhood
		true_joint_state = self.env.currstate
		consistent = False
		sample_count = 0
		rs = np.zeros(params['env']['n_agents'])
		while consistent is False and sample_count <= self.sample_count:
			sampled_joint_state = self.sampleJointState()
			for agent in range(params['env']['n_agents']):
				# rejection sampling
				rs[agent] = 1.0 * (true_joint_state[agent] in neighborhood[
					agent])  # agent accepts if the corresponding joint-state component is in its true state's neighbourhood
			if np.prod(rs) == 1.0:
				consistent = True
				if not consistent:
					consistent = False
					break
			sample_count += 1
		
		return sampled_joint_state