from optionCritic.option import createOptions
from doc import DOC
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning, AgentQLearning
# from distributed.belief import MultinomialDirichletBelief
from distributed.factored_belief_agent import MultinomialDirichletBelief
from optionCritic.gradients import TerminationGradient, IntraOptionGradient
import numpy as np
from utils.viz import plotReward, calcErrorInBelief, calcCriticValue, calcActionCriticValue, calcAgentActionValue, calcOptionValue
from utils.misc import saveModelandMetrics
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle
import os


class Trainer(object):
	def __init__(self, env, expt_folder):
		self.expt_folder = expt_folder
		self.env = env
		self.n_agents = params['env']['n_agents']
		self.writer = SummaryWriter(log_dir=expt_folder)
	
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
	
	# def calcAverageDurationFromEpisode(self, listOfOptions, numAgents):
	# 	agentOptions = {k: [item[k] for item in listOfOptions] for k in range(numAgents)}
	#
	# 	avg_dur = []
	# 	count = {k: 0 for k in range(numAgents)}
	# 	for k in list(agentOptions.keys()):
	# 		# print(k)
	# 		count[k] = 0
	# 		for i in range(len(agentOptions[k][:-1])):
	# 			if agentOptions[k][i] != agentOptions[k][i + 1]:
	# 				count[k] += 1
	# 		avg_dur.append(count[k] / (len(listOfOptions) - 1))
	# 	return avg_dur
		
	def train(self, pbar="default_pbar"):
		self.all_run_ep_steps = np.zeros((params['train']['n_runs'], params['train']['n_episodes']))
		self.all_run_ep_cum_rew = np.zeros((params['train']['n_runs'], params['train']['n_episodes']))
		for run in range(params['train']['n_runs']):
			self.run = run
			# put the agents to the same initial joint state as long as the random seed set in params['train'][
			# 'seed'] in modelConfig remains unchanged
			#joint_state = self.env.reset()
			
			
			# alpha = 0.001 * np.ones(len(self.env.states_list))


			alpha = 0.001 * np.ones(len(self.env.cell_list))
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
													   lr=params['train']['lr_action_critic'],
													   terminations=terminations,
													   qbigomega=self.critic)
			
			self.agent_q = AgentQLearning(discount=params['env']['discount'],
									 lr=params['train']['lr_agent_q'],
									 options=self.options)
			
			self.termination_gradient = TerminationGradient(self.options, self.critic, terminations)
			self.intra_option_policy_gradient = IntraOptionGradient(self.options)

			self.trainEpisodes()

			self.total_num_frames = params['train']['n_runs']*params['train']['n_episodes']*params['env']['episode_length']

			if pbar is not None:
				pbar = tqdm()

		return self.all_run_ep_steps, self.all_run_ep_cum_rew
			

	def trainEpisodes(self):

		iterations = 0
		# episode_critic_Q = []
		# episode_action_critic_Q = []

		params['policy']['temperature'] = 1.0
		cum_reward = 0.
		explore_param = 3*params['env']['goal_reward']

		for episode in range(params['train']['n_episodes']):
			#print('Episode : ', episode)
			
			params['policy']['temperature'] = 1
			if params['policy']['temperature'] > 0.1:
				params['policy']['temperature'] -= 0.025
			else:
				params['policy']['temperature'] = 0.01

			# if cum_reward/(episode+1) > explore_param and params['policy']['temperature'] > 0.1:
			# 	params['policy']['temperature'] -= 0.025
			# elif cum_reward/(episode+1) > explore_param and params['policy']['temperature'] < 0.1:
			# 	params['policy']['temperature'] = 0.01

			# if episode%10 == 0.:
			# 	params['policy']['temperature'] = 0.01
			
			# # put the agents to the same initial joint state as long as the random seed set in params['train'][
			# # 'f'] in modelConfig remains unchanged
			joint_state = self.env.reset()
			#print('Initial State:',joint_state)
			prev_joint_state = joint_state
			prev_joint_obs = [(joint_state[i],None) for i in range(self.env.n_agents)]
			prev_joint_action = tuple([None for _ in range(self.env.n_agents)])
			
			#
			# belief = MultinomialDirichletBelief(self.env, joint_observation)
			sampled_joint_state = joint_state
			old_feasible_states = list(joint_state)
			
			
			# d. Choose joint-option o based on softmax option-policy mu
			joint_option = self.doc.initializeOption(joint_state=joint_state)

			# joint action
			joint_action = self.doc.chooseAction()

			self.critic.start(joint_state, joint_option)
			self.action_critic.start(joint_state,joint_option,joint_action)
			self.agent_q.start(joint_state, joint_option, joint_action)
			
			# done = False
			cum_reward = 0
			itr_critic_Q = []
			itr_action_critic_Q = []
			c = 0.0
			
			for iteration in range(params['env']['episode_length']):
				
				# if iteration > 50 and iteration % 100 == 0:
				# 	if params['policy']['temperature'] > 0.1:
				# 		params['policy']['temperature'] -= 0.05
				# 	else:
				# 		params['policy']['temperature'] = 0.05

				# if iteration % 50 == 0:
				# 	print('Iteration : ', iteration, 'Cumulative Reward : ', cum_reward, 'Discovered Goals :', self.env.discovered_goals)


				joint_action = self.doc.chooseAction()

				reward, next_joint_state, done, _ = self.env.step(joint_action)

				# if reward > 0 and params['policy']['temperature'] > 0.05:
				# 	params['policy']['temperature'] -= 0.025

				reward += c
				
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

				#print('broadcasts',broadcasts,'error_tuple',selfishness_penalties)
				#print('bp', np.sum([broadcasts[i] * self.env.broadcast_penalty for i in range(len(broadcasts))]), 'sp', np.sum((1-broadcasts[i])*selfishness_penalties[i] for i in range(len(broadcasts))))
				
				reward += np.sum([broadcasts[i] * self.env.broadcast_penalty + (1-broadcasts[i])*selfishness_penalties[i] for i in range(len(broadcasts))])


				cum_reward = reward + params['env']['discount'] * cum_reward


				# if reward > 0.:
				# 	print('reward',reward,'cum_reward',cum_reward)
				#

				sampled_joint_state = [0 for _ in range(params['env']['n_agents'])]
				if iteration == 0:
					#print('Hi')
					joint_observation = prev_joint_obs
				else:
					joint_observation = self.env.get_observation(broadcasts)
				#print('iteration', iteration, 'brd', broadcasts, 'joint_obs', joint_observation)

				for i in range(params['env']['n_agents']):
					if iteration == 0 or broadcasts[i] == 1:
						#print('Hi')
						sampled_joint_state[i] = int(joint_state[i])
					else:
						#print('joint_observation[i]', joint_observation[i], 'old_feasible_states[i]', old_feasible_states[i])
						self.belief.update(joint_observation[i], old_feasible_states[i])
						sampled_joint_state[i] = int(self.belief.sampleJointState())

				while len(set(sampled_joint_state)) < len(sampled_joint_state):
					#print('Hi')
					resampling = {(k,v):False for k,v in zip(range(params['env']['n_agents']),sampled_joint_state)}
					for j in range(params['env']['n_agents']):
						if sampled_joint_state.count(sampled_joint_state[j]) > 1 and resampling[(j,sampled_joint_state[j])] is False:
							sampled_joint_state[j] = int(self.belief.sampleJointState())
							resampling[(j,sampled_joint_state[j])] = True
				#print('resampled', sampled_joint_state)

				js = [self.env.tocellcoord[s] for s in joint_state]
				sampled_js = [self.env.tocellcoord[s] for s in sampled_joint_state]
				#print('joint_state',js , 'sampled_joint_state', sampled_js)
				# if iteration % 50 == 0:
				#
				# 	print('ep: ', episode, 'iter', iteration, 'brd: ', broadcasts, 'obs: ', joint_observation, 'j_state: ', js, \
				#   'sampled_state: ', sampled_js)


				estimated_next_joint_state = self.estimate_next_joint_state(joint_observation,sampled_joint_state)
				
				# critic evaluation
				critic_feedback = self.doc.evaluateOption(critic=self.critic,
													 action_critic=self.action_critic,

													 agent_q = self.agent_q,

													 joint_state=estimated_next_joint_state, # this should be sampled_next_joint_state, s'_k in the algo

													 joint_option=joint_option,
													 joint_action=joint_action,
													 reward=reward,
													 done=done,
													 baseline=False)
				
				
				# xi A
				self.doc.improveOption(policy_obj=self.intra_option_policy_gradient,
								  termination_obj=self.termination_gradient,

								  sampled_joint_state=sampled_joint_state,
								  next_joint_state= next_joint_state,
								  estimated_next_joint_state =estimated_next_joint_state,

								  joint_option=joint_option,
								  joint_action=joint_action,
								  critic_feedback=critic_feedback
								   )
				
				# xi B

				next_joint_option, switch = self.doc.chooseOptionOnTermination(self.options, joint_option, sampled_joint_state)

				joint_option = next_joint_option
				
				prev_joint_state = joint_state
				joint_state = next_joint_state

				prev_joint_obs = joint_observation


				prev_joint_action = joint_action

				for i in range(params['env']['n_agents']):
					if iteration == 0:
						old_feasible_states[i] = self.belief.new_feasible_state(old_feasible_states[i],joint_observation[i])
					else:
						old_feasible_states[i] = self.belief.new_feasible_state(old_feasible_states[i])

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
				
				# optionValues = calcAgentActionValue(self.options)

				for agent in range(params['env']['n_agents']):
					for idx, option in enumerate(self.options[agent]):
						self.writer.add_scalar('agent'+str(agent)+'_option'+str(idx), calcOptionValue(option.policy.weights), iterations)
			
				# itr_critic_Q.append(critic_Q)
				# itr_action_critic_Q.append(action_critic_Q)

				# if iteration == 1:
				# 	break
			self.all_run_ep_steps[self.run,episode] = iteration
			self.all_run_ep_cum_rew[self.run,episode] = cum_reward
			# tensorboard plots
			self.writer.add_scalar('cumulative_reward', cum_reward, episode)
			self.writer.add_scalar('episode_length', iteration, episode)
			# self.writer.add_scalar('Critic_Q_episode', np.mean(itr_critic_Q), episode)
			self.writer.add_scalar('Critic_Q_episode', critic_Q, episode)
			# self.writer.add_scalar('Action_Critic_Q', np.mean(itr_action_critic_Q), episode)

			# Save model
			if episode == params['train']['n_episodes'] - 1:
				saveModelandMetrics(self)
			elif episode % 5 == 0:
				saveModelandMetrics(self)

			#TODO: Plot in average duration tensorboard





	
		
	