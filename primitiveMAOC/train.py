from optionCritic.option import createOptions
from doc import DOC
from modelConfig import params, paths
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning, AgentQLearning
from distributed.belief import MultinomialDirichletBelief
from optionCritic.gradients import TerminationGradient, IntraOptionGradient
import numpy as np
from utils.viz import plotReward, calcErrorInBelief, calcCriticValue, calcActionCriticValue, calcAgentActionValue, calcOptionValue
from utils.misc import saveModelandMetrics
from tensorboardX import SummaryWriter
import os



class Trainer(object):
	# def __init__(self, env, expt_folder):
	# 	self.expt_folder = expt_folder
	# 	self.env = env
	# 	self.n_agents = params['env']['n_agents']
	# 	self.writer = SummaryWriter(log_dir=expt_folder)
	# 	self.cum_reward_from_episodes = []
	# 	self.steps_from_episode = np.zeros((params['train']['n_runs'],params['train']['n_episodes']))
	# 	self.cum_rew_from_episode = np.zeros((params['train']['n_runs'],params['train']['n_episodes']))
	# 	self.critic_from_episode = np.zeros((params['train']['n_runs'],params['train']['n_episodes']))
	# 	self.avg_dur_from_episode = np.zeros((params['train']['n_runs'],params['train']['n_episodes'], params['env']['n_agents']))
		
	def __init__(self, env, expt_folder, timestr):
		self.expt_folder = expt_folder
		self.env = env
		self.n_agents = params['env']['n_agents']
		self.writer = SummaryWriter(log_dir=expt_folder)
		self.timestr = timestr

	def estimate_next_joint_state(self, joint_observation, sampled_joint_state):
		sampled_joint_state = tuple(np.sort(sampled_joint_state))
		res = np.zeros(len(joint_observation))
		for i in range(len(joint_observation)):
			if joint_observation[i] is not None:
				if joint_observation[i][1] is None:
					idx = np.random.choice(len(self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])))
					chosen_cell = self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])[idx]
					res[i] = self.env.tocellnum[chosen_cell]

				else:
					if self.env.occupancy[tuple(self.env.tocellcoord[joint_observation[i][0]] + self.env.directions[
						joint_observation[i][1]])] == 1:
						idx = np.random.choice(
							len(self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])))
						chosen_cell = self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])[idx]
						res[i] = self.env.tocellnum[chosen_cell]
					else:
						res[i] = self.env.tocellnum[tuple(self.env.tocellcoord[joint_observation[i][0]] +
														  self.env.directions[joint_observation[i][1]])]

			else:
				idx = np.random.choice(len(self.env.empty_adjacent(self.env.tocellcoord[sampled_joint_state[i]])))
				chosen_cell = self.env.empty_adjacent(self.env.tocellcoord[sampled_joint_state[i]])[idx]
				res[i] = self.env.tocellnum[chosen_cell]

		res = tuple([int(r) for r in res])
		return res
	

	# def train(self):
	# 	for _ in range(params['train']['n_runs']):
	# 		# put the agents to the same initial joint state as long as the random seed set in params['train'][
	# 		# 'seed'] in modelConfig remains unchanged
	# 		#joint_state = self.env.reset()

	# 		#self.run = run
			
			
	# 		alpha = 0.001 * np.ones(len(self.env.states_list))
	# 		self.belief = MultinomialDirichletBelief(self.env, alpha)

	# 		# deliberation cost 

	# 		# eta = params['train']['deliberation_cost']
		
			
	# 		# create option pool
	# 		self.options, self.mu_policy = createOptions(self.env)
	# 		# options is a list of option object. Each option object has its own termination policy and pi_policy.
	# 		# pi_policy for option 0 can be called as	:	options[0].policy.weights
	# 		# options[0].policy is the object of SoftmaxActionPolicy()
	# 		# termination for option 0 can be called as	:	options[0].termination.weights

	# 		#ption_policies = [option.policy for option in self.options]
	# 		#option_policies.extend([FixedActionPolicies(i, params['agent']['n_actions'])] for i in range(params['agent']['n_actions']))
			
	# 		# terminations = [option.termination for option in self.options]
	# 		# terminations.extend([OneStepTermination() for _ in range(params['agent']['n_actions'])])
	# 		terminations = []
	# 		for agent_idx in range(params['env']['n_agents']):
	# 			terminations.append([option.termination for option in self.options[agent_idx]])
	# 		#terminations = [OneStepTermination() for _ in range(params['agent']['n_options'])]
			
			
			
	# 		self.doc = DOC(self.env, self.options, self.mu_policy)
			
	# 		# # d. Choose joint-option o based on softmax option-policy mu
	# 		# joint_option = self.doc.initializeOption(joint_state=joint_state)
			
	# 		# # make the elected options unavailable
	# 		# for option in joint_option:
	# 		# 	self.options[option].available = False
			
	# 		# joint action
	# 		# joint_action = self.doc.chooseAction()
			
	# 		self.critic = IntraOptionQLearning(discount=params['env']['discount'],
	# 									  lr=params['train']['lr_critic'],
	# 									  terminations=terminations,
	# 									  weights=self.mu_policy.weights)
			
	# 		self.action_critic = IntraOptionActionQLearning(discount=params['env']['discount'],
	# 												   lr=params['train']['lr_action_critic'],
	# 												   terminations=terminations,
	# 												   qbigomega=self.critic)
			
	# 		self.agent_q = AgentQLearning(discount=params['env']['discount'],
	# 								 lr=params['train']['lr_agent_q'],
	# 								 options=self.options)
			
	# 		self.termination_gradient = TerminationGradient(self.options, self.critic)
	# 		self.intra_option_policy_gradient = IntraOptionGradient(self.options)

		
			
	# 		# for _ in range(params['train']['n_episodes']):
	# 		self.trainEpisodes()
			# self.steps_from_episode[run,episode], self.cum_rew_from_episode[run,episode], self.critic_from_episode[run,episode], self.avg_dur_from_episode[run,episode] =\
			#  iteration, cum_reward, calcCriticValue(self.critic.weights), np.array(switch_agent)/iteration

		
		# # Plots to report	
		# avg_len_epi = np.mean(self.steps_from_episode, axis=0)
		# avg_cum_rew = np.mean(self.cum_rew_from_episode, axis=0)
		# avg_crtic = np.mean(self.critic_from_episode, axis=0)
		# avg_dur = np.mean(self.avg_dur_from_episode, axis=0)
			

	def trainEpisodes(self):
		
		alpha = 0.001 * np.ones(len(self.env.states_list))
		self.belief = MultinomialDirichletBelief(self.env, alpha)

		# deliberation cost

		# eta = params['train']['deliberation_cost']

		# create option pool
		self.options, self.mu_policy = createOptions(self.env)
		# options is a list of option object. Each option object has its own termination policy and pi_policy.
		# pi_policy for option 0 can be called as	:	options[0].policy.weights
		# options[0].policy is the object of SoftmaxActionPolicy()
		# termination for option 0 can be called as	:	options[0].termination.weights

		terminations = []
		for agent_idx in range(params['env']['n_agents']):
			terminations.append([option.termination for option in self.options[agent_idx]])

		self.doc = DOC(self.env, self.options, self.mu_policy)

		self.critic = IntraOptionQLearning(discount=params['env']['discount'],
										   lr=params['train']['lr_critic'],
										   terminations=terminations,
										   weights=self.mu_policy.weights)

		self.action_critic = IntraOptionActionQLearning(discount=params['env']['discount'],
														lr=params['train']['lr_critic'],
														terminations=terminations,
														qbigomega=self.critic)

		self.agent_q = AgentQLearning(discount=params['env']['discount'],
									  lr=params['train']['lr_agent_q'],
									  options=self.options)

		self.termination_gradient = TerminationGradient(self.options, self.critic, terminations)
		self.intra_option_policy_gradient = IntraOptionGradient(self.options)

		# end of initialization

		iterations = 0
		# episode_critic_Q = []
		# episode_action_critic_Q = []

		params['policy']['temperature'] = 1
		# cum_reward = 0
		history = np.zeros((5, params['train']['n_episodes']), dtype=np.float32) # 0.Return 
		for episode in range(params['train']['n_episodes']):
			print('Episode : ', episode)
			

			# params['policy']['temperature'] = 1
			if params['policy']['temperature'] > 0.1:
				params['policy']['temperature'] -= 0.025
			else:
				params['policy']['temperature'] = 0.01
			# # put the agents to the same initial joint state as long as the random seed set in params['train'][
			# # 'f'] in modelConfig remains unchanged
			joint_state = self.env.reset()
			prev_joint_state = joint_state
			prev_joint_obs = [(joint_state[i],None) for i in range(self.env.n_agents)]
			prev_joint_action = tuple([None for _ in range(self.env.n_agents)])
			
			#
			# belief = MultinomialDirichletBelief(self.env, joint_observation)
			sampled_joint_state = joint_state
			old_feasible_states = joint_state
			
			
			# d. Choose joint-option o based on softmax option-policy mu
			joint_option = self.doc.initializeOption(joint_state=joint_state)

			# # make the elected options unavailable
			# for option in joint_option:
			# 	self.options[option].available = False

			# joint action
			joint_action = self.doc.chooseAction()

			self.critic.start(joint_state, joint_option)
			self.action_critic.start(joint_state,joint_option,joint_action)
			self.agent_q.start(joint_state, joint_option, joint_action)
			
			# termination_gradient = TerminationGradient(terminations, critic)
			# intra_option_policy_gradient = IntraOptionGradient(pi_policies)
			
			# done = False
			cum_reward = 0.
			# itr_reward = []
			# belief_error = []
			# options_episode = []
			# switch_agent = np.zeros(params['env']['n_agents'])

			c = 0.0
			
			for iteration in range(params['env']['episode_length']):

				if iteration % 50 == 0:
					print('Iteration : ', iteration, 'Cumulative Reward : ', cum_reward, 'Discovered Goals :', self.env.discovered_goals)


				
				

				#options_episode.append(joint_option)

				# iv
				joint_action = self.doc.chooseAction()
				
				# for agent in self.env.agents:
				# 	agent.action = joint_action[agent.ID]
				
				# v
				reward, next_joint_state, done, _ = self.env.step(joint_action)
				# reward = reward + c

				# cum_reward += reward
				
				# vi - absorbed in broadcastBasedOnQ function of Broadcast class
				
				# vii - viii
				broadcasts, selfishness_penalties = self.doc.toBroadcast(curr_true_joint_state = joint_state, 
											 prev_sampled_joint_state = sampled_joint_state,
											 prev_joint_obs = prev_joint_obs,
											 prev_true_joint_state = prev_joint_state, 
											 prev_joint_action = prev_joint_action, 
											 joint_option = joint_option, 
											 done = done, 
											 critic = self.critic, 
											 reward = reward)

				
				
				b = [broadcasts[i] * self.env.broadcast_penalty + (1-broadcasts[i])*selfishness_penalties[i] for i in range(len(broadcasts))]
				
				broadcast_penalty = np.sum(b)

				reward += broadcast_penalty

				cum_reward = reward + params['env']['discount'] * cum_reward
				
				# ix

				if iteration == 0:
					joint_observation = prev_joint_obs
				else:
					joint_observation = self.env.get_observation(broadcasts)
					self.belief.update(joint_observation, old_feasible_states)
					sampled_joint_state = self.belief.sampleJointState()  # iii

				estimated_next_joint_state = self.estimate_next_joint_state(joint_observation,sampled_joint_state)
				
				# x - critic evaluation
				critic_feedback = self.doc.evaluateOption(critic=self.critic,
													 action_critic=self.action_critic,
													 joint_state=estimated_next_joint_state, # this should be sampled_next_joint_state, s'_k in the algo
													 joint_option=joint_option,
													 joint_action=joint_action,
													 reward=reward,
													 done=done,
													 baseline=False)
				
				
				# xi A
				self.doc.improveOptionPrimitive(policy_obj=self.intra_option_policy_gradient,
								  sampled_joint_state=sampled_joint_state,
								  next_joint_state= next_joint_state,
								  estimated_next_joint_state =estimated_next_joint_state,
								  joint_option=joint_option,
								  joint_action=joint_action,
								  critic_feedback=critic_feedback
								   )
				
				# xi B

				next_joint_option, switch = self.doc.chooseOptionOnTermination(self.options, joint_option,
																			   sampled_joint_state) #TODO: should condition on sampled joint state
				#switches += switch
				# change_in_options = [currJO != nextJO for (currJO,nextJO) in zip(joint_option,next_joint_option)]

				# if switch:
				# 	c = 0.0001*params['train']['deliberation_cost']*switch
				# else:
				# 	c = 0

				# for i in range(params['env']['n_agents']):
				# 	if joint_option[i]!=next_joint_option[i]:
				# 		switch_agent[i] += 1
				

				# belief_error_step = calcErrorInBelief(self.env, joint_state, sampled_joint_state)
				# belief_error.append(belief_error_step)


				joint_option = next_joint_option
				
				prev_joint_state = joint_state
				joint_state = next_joint_state

				prev_joint_obs = joint_observation


				prev_joint_action = joint_action
				
				# self.belief.update(joint_observation,old_feasible_states)
				# sampled_joint_state = self.belief.sampleJointState() # iii
				
				# true_state_tocells = ([self.env.tocellcoord[s] for s in joint_state])
				# sampled_state_tocells = ([self.env.tocellcoord[s] for s in sampled_joint_state])
				# print('true joint state : ', true_state_tocells, 'sampled joint state :', sampled_state_tocells)
				

				old_feasible_states = self.belief.new_feasible_state(old_feasible_states,joint_observation)
				
				
				if done:
					break

				critic_Q, supNormQ = calcCriticValue(self.critic.weights)
				action_critic_Q = calcActionCriticValue(self.action_critic.weights)

				# tensorboard plots
				iterations += 1
				self.writer.add_scalar('reward_in_iteration', cum_reward, iterations)
				self.writer.add_scalar('broadcast_in_iteration', np.sum(broadcasts), iterations)
				self.writer.add_scalar('Critic_Q_itr', critic_Q, iterations)
				self.writer.add_scalar('Critic_Q_supNorm', supNormQ, iterations)
				self.writer.add_scalar('Action_Critic_Q-itr', action_critic_Q, iterations)

				for agent in range(params['env']['n_agents']):
					for idx, option in enumerate(self.options[agent]):
						self.writer.add_scalar('agent'+str(agent)+'_option'+str(idx), calcOptionValue(option.policy.weights), iterations)
			

			# self.steps_from_episode[self.run,episode] = iteration
			# self.cum_rew_from_episode[self.run,episode] = cum_reward
			# self.critic_from_episode[self.run,episode] = calcCriticValue(self.critic.weights)
			# self.avg_dur_from_episode[self.run,episode] = np.array(switch_agent)/iteration
# tensorboard plots
			self.writer.add_scalar('cumulative_reward', cum_reward, episode)
			self.writer.add_scalar('episode_length', iteration, episode)
			# self.writer.add_scalar('Critic_Q_episode', np.mean(itr_critic_Q), episode)
			self.writer.add_scalar('Critic_Q_episode', critic_Q, episode)
			# self.writer.add_scalar('Action_Critic_Q', np.mean(itr_action_critic_Q), episode)

			# save
			history[0, episode] = cum_reward
			history[1, episode] = iteration
			history[2, episode] = critic_Q
			history[3, episode] = action_critic_Q
			history[4, episode] = supNormQ

			# print(cum_reward, iteration, critic_Q, action_critic_Q)

			np.save(os.path.join(self.expt_folder, 'history.npy'), history)
			np.save(os.path.join(paths['output']['graphs_folder'], 'history_%s.npy'%self.timestr), history)
			# np.save(os.path.join('/private/home/sumanab/multiagent/experiments/', self.expt_folder, 'history_%s.npy' % self.timestr), history)

			# self.output.append(history)

			# Save model
			if episode == params['train']['n_episodes'] - 1:
				saveModelandMetrics(self)
			elif episode % 10 == 0:
				saveModelandMetrics(self)


	# def estimate_next_joint_state(self, joint_observation, sampled_joint_state):
	# 	sampled_joint_state = tuple(np.sort(sampled_joint_state))
	# 	res = np.zeros(len(joint_observation))
	# 	for i in range(len(joint_observation)):
	# 		if joint_observation[i] is not None:
	# 			if joint_observation[i][1] is None:
	# 				idx = np.random.choice(len(self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])))
	# 				chosen_cell = self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])[idx]
	# 				res[i] = self.env.tocellnum[chosen_cell]

	# 			else:
	# 				if self.env.occupancy[tuple(self.env.tocellcoord[joint_observation[i][0]] + self.env.directions[
	# 					joint_observation[i][1]])] ==1:
	# 					idx = np.random.choice(len(self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])))
	# 					chosen_cell = self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])[idx]
	# 					res[i] = self.env.tocellnum[chosen_cell]
	# 				else:
	# 					res[i] = self.env.tocellnum[tuple(self.env.tocellcoord[joint_observation[i][0]] +
	# 												self.env.directions[joint_observation[i][1]])]
					
	# 		else:
	# 			idx = np.random.choice(len(self.env.empty_adjacent(self.env.tocellcoord[sampled_joint_state[i]])))
	# 			chosen_cell = self.env.empty_adjacent(self.env.tocellcoord[sampled_joint_state[i]])[idx]
	# 			res[i] = self.env.tocellnum[chosen_cell]
		
	# 	res = tuple([int(r) for r in res])
	# 	return res


	# def calcAverageDurationFromEpisode(self, listOfOptions,numAgents):
	# 	agentOptions = {k: [item[k] for item in listOfOptions] for k in range(numAgents)}

	# 	avg_dur = []
	# 	count = {k:0 for k in range(numAgents)}
	# 	for k in list(agentOptions.keys()):
	# 	    #print(k)
	# 	    count[k] = 0
	# 	    for i in range(len(agentOptions[k][:-1])):
	# 	        if agentOptions[k][i] != agentOptions[k][i+1]:
	# 	            count[k] += 1
	# 	    avg_dur.append(count[k]/(len(listOfOptions)-1))
	# 	return avg_dur




	
		
	