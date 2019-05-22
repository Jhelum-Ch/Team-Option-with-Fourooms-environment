from optionCritic.option import createOptions
from doc import DOC
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning, AgentQLearning
from distributed.belief import MultinomialDirichletBelief
from optionCritic.gradients import TerminationGradient, IntraOptionGradient
import numpy as np
from utils.viz import plotReward, calcErrorInBelief, calcCriticValue, calcActionCriticValue, calcAgentActionValue, calcOptionValue
from utils.misc import saveModelandMetrics
from tensorboardX import SummaryWriter
import pickle
import os
import threading


class Trainer(object):
	def __init__(self, env, expt_folder):
		self.expt_folder = expt_folder
		self.env = env
		self.n_agents = params['env']['n_agents']
		self.writer = SummaryWriter(log_dir=expt_folder)
		self.output = []
	
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
		
	def train(self):
		threads = []
		seeds = np.random.randint(5000, size=params['train']['n_runs'])
		for _, (run, curr_seed) in enumerate(zip(range(params['train']['n_runs']), seeds)):
			print('run:',run)
			t = threading.Thread(self.trainEpisodes(curr_seed))
			threads.append(t)
			t.start()

		for x in threads:
			x.join()

		np.save(os.path.join(self.expt_folder, 'history_all_runs.npy'), np.asarray(self.output))
		np.save(os.path.join(self.expt_folder, 'seeds.npy'), np.asarray(seeds))
			

	def trainEpisodes(self, myseed):

		params['train']['seed'] = myseed
		print(myseed)


		# initialize everything

		# put the agents to the same initial joint state as long as the random seed set in params['train'][
		# 'seed'] in modelConfig remains unchanged
		# joint_state = self.env.reset()

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
		history = np.zeros((params['train']['n_episodes'], 4), dtype=np.float32) # 0.Return 1.episode_length 2.criticQValue, 3.action_critic_Q

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
			print('Initial State:',joint_state)
			prev_joint_state = joint_state
			prev_joint_obs = [(joint_state[i],None) for i in range(self.env.n_agents)]
			prev_joint_action = tuple([None for _ in range(self.env.n_agents)])
			
			#
			# belief = MultinomialDirichletBelief(self.env, joint_observation)
			sampled_joint_state = joint_state
			old_feasible_states = joint_state
			
			
			# d. Choose joint-option o based on softmax option-policy mu
			joint_option = self.doc.initializeOption(joint_state=joint_state)

			# joint action
			joint_action = self.doc.chooseAction()

			self.critic.start(joint_state, joint_option)
			self.action_critic.start(joint_state,joint_option,joint_action)
			self.agent_q.start(joint_state, joint_option, joint_action)
			
			# done = False
			cum_reward = 0
			c = 0.0
			
			for iteration in range(params['env']['episode_length']):
				
				# if iteration > 50 and iteration % 100 == 0:
				# 	if params['policy']['temperature'] > 0.1:
				# 		params['policy']['temperature'] -= 0.05
				# 	else:
				# 		params['policy']['temperature'] = 0.05

				if iteration % 50 == 0:
					print('Iteration : ', iteration, 'Cumulative Reward : ', cum_reward, 'Discovered Goals :', self.env.discovered_goals)


				joint_action = self.doc.chooseAction()

				reward, next_joint_state, done, _ = self.env.step(joint_action)

				if reward > 0 and params['policy']['temperature'] > 0.05:
					params['policy']['temperature'] -= 0.025

				reward += c
				
				# vi - absorbed in broadcastBasedOnQ function of Broadcast class
				
				# vii - viii
				broadcasts, error_tuple = self.doc.toBroadcast(curr_true_joint_state = joint_state,
											 prev_sampled_joint_state = sampled_joint_state,
											 prev_joint_obs = prev_joint_obs,
											 prev_true_joint_state = prev_joint_state,
											 prev_joint_action = prev_joint_action,
											 joint_option = joint_option,
											 done = done,
											 critic = self.critic,
											 reward = reward)

				b = [broadcasts[i] * self.env.broadcast_penalty + (1-broadcasts[i])*error_tuple[i] for i in range(len(broadcasts))]
				broadcast_penalty = np.sum(b)
				# print(self.env.broadcast_penalty, b, broadcast_penalty)
				reward += broadcast_penalty

				cum_reward = reward + params['env']['discount'] * cum_reward
				# if reward > 0.:
				# 	print('reward',reward,'cum_reward',cum_reward)

				if iteration == 0:
					joint_observation = prev_joint_obs
				else:
					joint_observation = self.env.get_observation(broadcasts)
					self.belief.update(joint_observation, old_feasible_states)
					sampled_joint_state = self.belief.sampleJointState()

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
				
				# optionValues = calcAgentActionValue(self.options)

				for agent in range(params['env']['n_agents']):
					for idx, option in enumerate(self.options[agent]):
						self.writer.add_scalar('agent'+str(agent)+'_option'+str(idx), calcOptionValue(option.policy.weights), iterations)
			
				# itr_critic_Q.append(critic_Q)
				# itr_action_critic_Q.append(action_critic_Q)

				# if iteration == 1:
				# 	break

			# tensorboard plots
			self.writer.add_scalar('cumulative_reward', cum_reward, episode)
			self.writer.add_scalar('episode_length', iteration, episode)
			# self.writer.add_scalar('Critic_Q_episode', np.mean(itr_critic_Q), episode)
			self.writer.add_scalar('Critic_Q_episode', critic_Q, episode)
			# self.writer.add_scalar('Action_Critic_Q', np.mean(itr_action_critic_Q), episode)

			# save
			history[episode, 0] = cum_reward
			history[episode, 1] = iteration
			history[episode, 2] = critic_Q
			history[episode, 3] = action_critic_Q

			# print(cum_reward, iteration, critic_Q, action_critic_Q)

			np.save(os.path.join(self.expt_folder, 'history%s.npy'%iteration), history)

			self.output.append(history)

			# # Save model
			# if episode == params['train']['n_episodes'] - 1:
			# 	saveModelandMetrics(self)
			# elif episode % 5 == 0:
			# 	saveModelandMetrics(self)

			#TODO: Plot in average duration tensorboard




	
		
	