from optionCritic.option import createOptions
from doc import DOC
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning, AgentQLearning
from optionCritic.policies import FixedActionPolicies
from optionCritic.termination import OneStepTermination
from distributed.belief import MultinomialDirichletBelief
from optionCritic.gradients import TerminationGradient, IntraOptionGradient
import numpy as np
from utils.viz import plotReward, calcErrorInBelief, calcCriticValue, calcActionCriticValue, calcAgentActionValue
from utils.misc import saveModelandMetrics
from tensorboardX import SummaryWriter


class Trainer(object):
	def __init__(self, env, expt_folder):
		self.expt_folder = expt_folder
		self.env = env
		self.n_agents = params['env']['n_agents']
		self.writer = SummaryWriter(log_dir=expt_folder)
		self.cum_reward_from_episodes = []
		self.steps_from_episode = np.zeros((params['train']['n_runs'],params['train']['n_episodes']))
		self.cum_rew_from_episode = np.zeros((params['train']['n_runs'],params['train']['n_episodes']))
		self.critic_from_episode = np.zeros((params['train']['n_runs'],params['train']['n_episodes']))
		self.avg_dur_from_episode = np.zeros((params['train']['n_runs'],params['train']['n_episodes'], params['env']['n_agents']))
		
	def train(self):
		for run in range(params['train']['n_runs']):
			# put the agents to the same initial joint state as long as the random seed set in params['train'][
			# 'seed'] in modelConfig remains unchanged
			#joint_state = self.env.reset()

			self.run = run
			
			
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

			option_policies = [option.policy for option in self.options]
			#option_policies.extend([FixedActionPolicies(i, params['agent']['n_actions'])] for i in range(params['agent']['n_actions']))
			
			# terminations = [option.termination for option in self.options]
			# terminations.extend([OneStepTermination() for _ in range(params['agent']['n_actions'])])
			terminations = [OneStepTermination() for _ in range(params['agent']['n_options'])]
			
			
			self.doc = DOC(self.env, self.options, self.mu_policy)
			
			# # d. Choose joint-option o based on softmax option-policy mu
			# joint_option = self.doc.initializeOption(joint_state=joint_state)
			
			# # make the elected options unavailable
			# for option in joint_option:
			# 	self.options[option].available = False
			
			# joint action
			# joint_action = self.doc.chooseAction()
			
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
			
			self.termination_gradient = TerminationGradient(self.options, self.critic)
			self.intra_option_policy_gradient = IntraOptionGradient(self.options)

		
			
			# for _ in range(params['train']['n_episodes']):
			self.trainEpisodes()
			# self.steps_from_episode[run,episode], self.cum_rew_from_episode[run,episode], self.critic_from_episode[run,episode], self.avg_dur_from_episode[run,episode] =\
			#  iteration, cum_reward, calcCriticValue(self.critic.weights), np.array(switch_agent)/iteration

		
		# Plots to report	
		avg_len_epi = np.mean(self.steps_from_episode, axis=0)
		avg_cum_rew = np.mean(self.cum_rew_from_episode, axis=0)
		avg_crtic = np.mean(self.critic_from_episode, axis=0)
		avg_dur = np.mean(self.avg_dur_from_episode, axis=0)
			

	def trainEpisodes(self):
		
		sum_of_rewards_per_episode = []
		episode_length = []
		avg_belief_error = []
		iterations = 0
		switches = 0
		# avg_dur_from_episode = []
		# cum_rew_from_episode = []

		for episode in range(params['train']['n_episodes']):
			print('Episode : ', episode)
			
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
			cum_reward = 0
			itr_reward = []
			belief_error = []
			options_episode = []
			switch_agent = np.zeros(params['env']['n_agents'])

			#c = 0.0
			
			for iteration in range(params['env']['episode_length']):

				# print('Iteration : ', iteration, 'Cumulative Reward : ', cum_reward, 'Discovered Goals :',
				# 	  self.env.discovered_goals)

				if iteration > 50 and iteration % 20 == 0:
					if params['policy']['temperature'] > 0.1:
						params['policy']['temperature'] -= 0.1
					else:
						params['policy']['temperature'] = 0.01

				if iteration % 50 == 0:
					print('Iteration : ', iteration, 'Cumulative Reward : ', cum_reward, 'Discovered Goals :', self.env.discovered_goals)

				options_episode.append(joint_option)

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
				broadcasts, error_tuple = self.doc.toBroadcast(curr_true_joint_state = joint_state, 
											 prev_sampled_joint_state = sampled_joint_state,
											 prev_joint_obs = prev_joint_obs,
											 prev_true_joint_state = prev_joint_state, 
											 prev_joint_action = prev_joint_action, 
											 joint_option = joint_option, 
											 done = done, 
											 critic = self.critic, 
											 reward = reward)

				
				
				reward += np.sum([broadcasts[i] * self.env.broadcast_penalty + (1-broadcasts[i])*error_tuple[i] for i in range(len(broadcasts))])

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
				switches += switch
				# change_in_options = [currJO != nextJO for (currJO,nextJO) in zip(joint_option,next_joint_option)]

				# if switch:
				# 	c = 0.0001*params['train']['deliberation_cost']*switch
				# else:
				# 	c = 0

				for i in range(params['env']['n_agents']):
					if joint_option[i]!=next_joint_option[i]:
						switch_agent[i] += 1
				

				belief_error_step = calcErrorInBelief(self.env, joint_state, sampled_joint_state)
				belief_error.append(belief_error_step)


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
				
				
				itr_reward.append(cum_reward)
				if not iteration%30 or done:
					plotReward(itr_reward,'iterations','cumulative reward',self.expt_folder,
							   'iteration_reward_'+ str(episode) +'.png')
					
					plotReward(belief_error, 'iterations', 'error', self.expt_folder,
							   'belief_error_' + str(episode) + '.png')
				
				if done:
					break

			self.steps_from_episode[self.run,episode] = iteration
			self.cum_rew_from_episode[self.run,episode] = cum_reward
			self.critic_from_episode[self.run,episode] = calcCriticValue(self.critic.weights)
			self.avg_dur_from_episode[self.run,episode] = np.array(switch_agent)/iteration
				#action_critic_Q = calcActionCriticValue(self.action_critic.weights)
				
				# # tensorboard plots
				# iterations += 1
				# self.writer.add_scalar('reward_in_iteration', cum_reward, iterations)
				# self.writer.add_scalar('broadcast_in_iteration', np.sum(broadcasts), iterations)
				# self.writer.add_scalar('option switches', switches, iterations)
				# self.writer.add_scalar('Critic_Q_itr', critic_Q, iterations)
				# self.writer.add_scalar('Action_Critic_Q-itr', action_critic_Q, iterations)
				# self.writer.add_scalar('Belief_Error_itr', belief_error_step, iterations)
				
				# optionValues = calcAgentActionValue(self.options)
				
				# for idx, option in enumerate(self.options):
				# 	self.writer.add_scalar('option '+str(idx)+str(optionValues[idx]), iterations)
					
			
			# cum_rew_from_episode.append(itr_reward)	
			# avg_rew_from_episode = list(np.mean(np.array(cum_rew_from_episode), axis = 0))

			sum_of_rewards_per_episode.append(itr_reward[-1])
			plotReward(sum_of_rewards_per_episode, 'episodes', 'sum of rewards', self.expt_folder,
					   'reward_per_episode.png')
			
			episode_length.append(len(itr_reward))
			print('episode length :', episode_length)
			plotReward(episode_length, 'episodes', 'length', self.expt_folder,
				   'episode_length.png')
			
			avg_belief_error.append(np.mean(belief_error))
			plotReward(avg_belief_error, 'episodes', 'mean_belief_error', self.expt_folder,
				   'mean_belief_error_per_episode.png')

			# avg_dur = self.calcAverageDurationFromEpisode(options_episode, len(joint_option))
			# avg_dur_from_episode.append(avg_dur)
			
			# tensorboard plots
			self.writer.add_scalar('cumulative_reward', cum_reward, episode)
			self.writer.add_scalar('episode_length', iteration, episode)
			# self.writer.add_scalar('episode_length', len(itr_reward), episode)
			#self.writer.add_scalar('mean_belief_error', np.mean(belief_error), episode)
			self.writer.add_scalar('Critic_Q_episode', calcCriticValue(self.critic.weights), episode)
			#self.writer.add_scalar('Action_Critic_Q', action_critic_Q, episode)
			self.writer.add_scalar('average_duration', np.array(switch_agent)/iteration, episode)
			
			# Save model
			saveModelandMetrics(self)

			#return (episode, iteration, cum_reward, calcCriticValue(self.critic.weights), np.array(switch_agent)/iteration)
			
			
			#TODO: save checkpoint


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
						joint_observation[i][1]])] ==1:
						idx = np.random.choice(len(self.env.empty_adjacent(self.env.tocellcoord[joint_observation[i][0]])))
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




	
		
	