import collections
import numpy as np
import scipy.stats as stats
import random
from modelConfig import params
import copy
from itertools import product
from fourroomsEnv import FourroomsMA
import collections


class MultinomialDirichletBelief:
	def __init__(self, env, alpha, sample_count=20):
		super(MultinomialDirichletBelief, self).__init__()
		self.env = copy.deepcopy(env)
		# self.env = env.deepcopy()
		self.alpha = alpha
		# self.joint_observation = joint_observation
		self.sample_count = sample_count  # sample_count is for rejection sampling
		#self.curr_joint_state = self.env.currstate
		self.states_list = self.env.states_list
		self.counts = collections.Counter(self.states_list)

	
	def update(self, joint_observation, old_feasible_states): #old_feasible_states is a list of integers/lists
		#assert isinstance(counts, dict), "counts is a disctionary"
		

		self.joint_observation = joint_observation
		
		new_feasible_states = self.new_feasible_state(old_feasible_states, self.joint_observation)
		self.old_feasible_states = new_feasible_states # update feasible states
		
		
		# Set the counts vector zero
		counts_vec = [self.counts.get(i, 0) for i in range(len(self.states_list))]

		if True not in [self.joint_observation[i] is None for i in range(len(self.joint_observation))]:
			observed_states = tuple([item[0] for item in self.joint_observation])

			for item in self.states_list:
				if item == observed_states:
					counts_vec[self.states_list.index(item)] += 10000000
					break
					
		else:
			b = self.new_feasible_state(self.old_feasible_states,self.joint_observation)
			list_final = []
			for i in range(len(self.joint_observation)):
				if self.joint_observation[i] is None:
					list_int = b[i]
					list_final.append(list_int)
				else:
					list_int = []
					list_int.append(int(self.joint_observation[i][0]))
					list_final.append(list_int)
			updated_states_list = list(product(*list_final))
			
			# print('removing duplicates:')
			# for joint_state in updated_states_list:
			# 	if len(joint_state) > len(set(joint_state)):
			# 		print(joint_state, ' removed')
			# 		updated_states_list.remove(joint_state)
			#
			# import pdb; pdb.set_trace()
			# print(updated_states_list)
			
			new_updated_states_list = []
			for joint_state in updated_states_list:
				if len(list(joint_state)) == len(set(list(joint_state))):
					new_updated_states_list.append(joint_state)
			
			# print('Entering updation:')
			for item in new_updated_states_list:
				# print(item)
				item = tuple([int(i) for i in item])
				self.counts[item] += 1000000
				counts_vec[self.states_list.index(item)] += 1000000


		return MultinomialDirichletBelief(self.env, np.add(self.alpha,counts_vec))

	def pmf(self):
		a = stats.dirichlet.rvs(self.alpha, size=1, random_state=1)
		return a[0]
		
	
	def sampleJointState(self):  # sample one joint_state from posterior
		#self.joint_observation = joint_observation
		sampled_state_idx = int(np.random.choice(range(len(self.states_list)), 1, p=self.pmf()))
		return self.states_list[sampled_state_idx]
	
	
	def rejectionSamplingNeighbour(self):
		
		# determine neighborhood of each agent
		neighborhood = np.empty((params['env']['n_agents'], params['agent']['n_actions']))  # create an empty n-d-array
		for agent in range(params['env']['n_agents']):
			for action in range(params['agent']['n_actions']):
				#self.env.currstate = self.curr_joint_state
				neighboring_state = self.env.neighbouringState(agent, action)
				neighborhood[agent, action] = neighboring_state
		#print('neighborhood', neighborhood)
		# each agent rejects a sample from common-belief posterior based on its own neighborhood
		consistent = False
		sample_count = 0
		rs = np.zeros(params['env']['n_agents'])
		while consistent is False and sample_count <= self.sample_count:
			sampled_joint_state = self.sampleJointState()
			print(sampled_joint_state)
			for agent in range(params['env']['n_agents']):
				# rejection sampling
				rs[agent] = 1.0 * (sampled_joint_state[agent] in neighborhood[
					agent])  # agent accepts if the corresponding joint-state component is in its true state's neighbourhood
			if np.prod(rs) == 1.0:
				consistent = True

			sample_count += 1
		
		return sampled_joint_state
	
	def estimated_feasible_state(self, agent_state, action=None):
		feasible_state_list = np.zeros(params['agent']['n_actions'])
		currcell = self.env.tocellcoord[agent_state]
		if action is None:
			for action in range(params['agent']['n_actions']):
				direction = self.env.directions[action]
				if self.env.occupancy[tuple(currcell + direction)] != 1:
					feasible_state_list[action] = self.env.tocellnum[tuple(currcell + direction)]
			return [item for item in feasible_state_list if item != 0.]
		else:
			direction = self.env.directions[action]
			if self.env.occupancy[tuple(currcell + direction)] == 1:
				next_state = self.env.tocellnum[tuple(currcell)]
			else:
				next_state = self.env.tocellnum[tuple(currcell + direction)]
			return next_state
	
	def new_feasible_state(self, old_feasible_states, obs):  # old_feasible_states can be either list of integers or
		# list of lists
		new_feasible_states = []
		
		for i in range(len((obs))):
			
			if obs[i] is not None:
				# print('obs',obs[i])
				# if obs[i][1] is not None:
				if self.env.occupancy[tuple(self.env.tocellcoord[obs[i][0]] + self.env.directions[obs[i][1]])] == 1:
					next_est_state = self.env.tocellnum[tuple(self.env.tocellcoord[obs[i][0]])]
				else:
					next_est_state = self.env.tocellnum[
						tuple(self.env.tocellcoord[obs[i][0]] + self.env.directions[obs[i][1]])]
				
				new_feasible_states.append(next_est_state)
			else:
				if isinstance(old_feasible_states[i], (int, np.integer)):
					new_list = self.estimated_feasible_state(old_feasible_states[i])  # new_list is list
					new_feasible_states.append(new_list)
				else:  # if isinstance(old_feasible_states[i],list)
					new_list = [self.estimated_feasible_state(s) for s in
								old_feasible_states[i]]  # new_list is list of list
					flatten_new_list = [int(s) for item in new_list for s in item]
					new_feasible_states.append(list(set(flatten_new_list)))
		
		return new_feasible_states

