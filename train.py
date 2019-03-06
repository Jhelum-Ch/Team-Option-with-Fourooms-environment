# # from optionCritic.option import createOptions
# # from doc import DOC
# # from modelConfig import params
# # from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning, AgentQLearning
# # from distributed.belief import MultinomialDirichletBelief
# # from optionCritic.gradients import TerminationGradient, IntraOptionGradient
# # import numpy as np
# # # import matplotlib
# # # matplotlib.use('Agg')
# # # import matplotlib.pyplot as plt
# # from utils.viz import plotReward


# # class Trainer(object):
# # 	def __init__(self, env, expt_folder):
# # 		self.expt_folder = expt_folder
# # 		self.env = env
# # 		self.n_agents = params['env']['n_agents']
	
# # 	def train(self):
# # 		for _ in range(params['train']['n_epochs']):
# # 			self.trainEpisode()
			
# # 	# def putAgentsToGrid(self):
# # 	# 	idx = 0
# # 	# 	for agent in self.env.agents:
# # 	# 		agent.state = params['env']['initial_joint_state'][idx]
# # 	# 		idx += 1
	
# # 	def trainEpisode(self):
		
# # 		episode_reward = []
# # 		for episode in range(params['train']['n_episodes']):
# # 			#print('Episode : ', episode)
# # 			# put the agents to the same initial joint state as long as the random seed set in params['train'][
# # 			# 'seed'] in modelConfig remains unchanged
# # 			joint_state = self.env.reset()
# # 			joint_observation = joint_state
			
# # 			belief = MultinomialDirichletBelief(self.env, joint_observation)
# # 			sampled_joint_state= joint_state
			
# # 			# create option pool
# # 			options, mu_policy = createOptions(self.env)
# # 			# options is a list of option object. Each option object has its own termination policy and pi_policy.
# # 			# pi_policy for option 0 can be called as	:	options[0].policy.weights
# # 			# options[0].policy is the object of SoftmaxActionPolicy()
# # 			# termination for option 0 can be called as	:	options[0].termination.weights
			
# # 			terminations = [option.termination for option in options]
# # 			pi_policies = [option.policy for option in options]
			
# # 			doc = DOC(self.env, options, mu_policy)
			
# # 			# d. Choose joint-option o based on softmax option-policy mu
# # 			joint_option = doc.initializeOption(joint_state=joint_state)
			
# # 			# make the elected options unavailable
# # 			for option in joint_option:
# # 				options[option].available = False
			
# # 			# joint action
# # 			joint_action = doc.chooseAction()
			
# # 			critic = IntraOptionQLearning(discount= params['train']['discount'],
# # 										  lr= params['train']['lr_critic'],
# # 										  terminations= terminations,
# # 										  weights= mu_policy.weights)
			
# # 			action_critic = IntraOptionActionQLearning(discount= params['train']['discount'],
# # 													   lr = params['train']['lr_action_critic'],
# # 													   terminations=terminations,
# # 													   qbigomega=critic)
			
# # 			agent_q = AgentQLearning(discount=params['train']['discount'],
# # 									 lr=params['train']['lr_agent_q'],
# # 									 options=options)
			
# # 			critic.start(joint_state, joint_option)
# # 			action_critic.start(joint_state,joint_option,joint_action)
# # 			agent_q.start(joint_state, joint_option, joint_action)
			
# # 			termination_gradient = TerminationGradient(terminations, critic)
# # 			intra_option_policy_gradient = IntraOptionGradient(pi_policies)
			
# # 			done = False
# # 			cum_reward = 0
# # 			itr_reward = []
# # 			for iteration in range(params['env']['episode_length']):
# # 				print('Iteration : ', iteration, 'Cumulative Reward : ', cum_reward)
# # 				# iv
# # 				joint_action = doc.chooseAction()

				
# # 				# v
# # 				reward, next_joint_state, done, _ = self.env.step(joint_action)
# # 				print('Episode: ', episode, 'done: ', done, 'discovered goals: ', self.env.discovered_goals)
				
				
# # 				# vi - absorbed in broadcastBasedOnQ function of Broadcast class
				
# # 				# vii - viii
# # 				broadcasts = doc.toBroadcast(next_true_joint_state=next_joint_state,
# # 											 sampled_curr_joint_state=sampled_joint_state,
# # 											 joint_option=joint_option,
# # 											 done=done,
# # 											 critic=critic,
# # 											 reward=reward)
# # 				print('Broadcasts', broadcasts)
				
# # 				cum_reward += reward + np.sum([i*self.env.broadcast_penalty for i in broadcasts])
# # 				# ix
# # 				next_joint_observation = self.env.get_observation(broadcasts)
				
# # 				# x - critic evaluation
# # 				critic_feedback = doc.evaluateOption(critic=critic,
# # 													 action_critic=action_critic,
# # 													 joint_state=joint_state,
# # 													 joint_option=joint_option,
# # 													 joint_action=joint_action,
# # 													 reward=reward,
# # 													 done=done,
# # 													 baseline=False)
				
				
# # 				# xi A
# # 				doc.improveOption(policy_obj=intra_option_policy_gradient,
# # 								  termination_obj=termination_gradient,
# # 								  joint_state=sampled_joint_state,
# # 								  joint_option=joint_option,
# # 								  joint_action=joint_action,
# # 								  critic_feedback=critic_feedback
# # 								   )
				
# # 				# xi B
# # 				joint_option = doc.chooseOptionOnTermination(options, joint_option, sampled_joint_state)
				
# # 				joint_state = next_joint_state
# # 				joint_observation = next_joint_observation
# # 				sampled_joint_state = belief.sampleJointState(joint_observation) # iii
				
# # 				itr_reward.append(cum_reward)
# # 				if not iteration%30:
# # 					plotReward(itr_reward,'iterations','cumulative reward',self.expt_folder, 'iteration_reward.png')
				
# # 				if done:
# # 					break
					
# # 			episode_reward.append(np.mean(itr_reward))
# # 			plotReward(episode_reward, 'episodes', 'cumulative reward', self.expt_folder, 'episode_reward.png')
	
	
# from optionCritic.option import createOptions
# from doc import DOC
# from modelConfig import params
# from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning, AgentQLearning
# from distributed.belief import MultinomialDirichletBelief
# from optionCritic.gradients import TerminationGradient, IntraOptionGradient
# import numpy as np
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# from utils.viz import plotReward


# class Trainer(object):
# 	def __init__(self, env, expt_folder):
# 		self.expt_folder = expt_folder
# 		self.env = env
# 		self.n_agents = params['env']['n_agents']
# 		#self.discount = params['env']['discount']
		
# 	def train(self):
# 		for _ in range(params['train']['n_runs']):
# 			# put the agents to the same initial joint state as long as the random seed set in params['train'][
# 			# 'seed'] in modelConfig remains unchanged
# 			joint_state = self.env.reset()
# 			joint_observation = joint_state
			
# 			self.belief = MultinomialDirichletBelief(self.env)
# 			# sampled_joint_state = joint_state
			
# 			# create option pool
# 			self.options, self.mu_policy = createOptions(self.env)
# 			# options is a list of option object. Each option object has its own termination policy and pi_policy.
# 			# pi_policy for option 0 can be called as	:	options[0].policy.weights
# 			# options[0].policy is the object of SoftmaxActionPolicy()
# 			# termination for option 0 can be called as	:	options[0].termination.weights
			
# 			terminations = [option.termination for option in self.options]
# 			pi_policies = [option.policy for option in self.options]
			
# 			self.doc = DOC(self.env, self.options, self.mu_policy)
			
# 			# # d. Choose joint-option o based on softmax option-policy mu
# 			# joint_option = self.doc.initializeOption(joint_state=joint_state)
			
# 			# # make the elected options unavailable
# 			# for option in joint_option:
# 			# 	self.options[option].available = False
			
# 			# joint action
# 			# joint_action = self.doc.chooseAction()
			
# 			self.critic = IntraOptionQLearning(discount=params['env']['discount'],
# 										  lr=params['train']['lr_critic'],
# 										  terminations=terminations,
# 										  weights=self.mu_policy.weights)
			
# 			self.action_critic = IntraOptionActionQLearning(discount=params['env']['discount'],
# 													   lr=params['train']['lr_action_critic'],
# 													   terminations=terminations,
# 													   qbigomega=self.critic)
			
# 			self.agent_q = AgentQLearning(discount=params['env']['discount'],
# 									 lr=params['train']['lr_agent_q'],
# 									 options=self.options)
			
# 			self.termination_gradient = TerminationGradient(terminations, self.critic)
# 			self.intra_option_policy_gradient = IntraOptionGradient(pi_policies)
			
# 			# for _ in range(params['train']['n_episodes']):
# 			self.trainEpisodes()
			

# 	def trainEpisodes(self):
		
# 		sum_of_rewards_per_episode = []
# 		for episode in range(params['train']['n_episodes']):
# 			print('Episode : ', episode)
# 			# # put the agents to the same initial joint state as long as the random seed set in params['train'][
# 			# # 'seed'] in modelConfig remains unchanged
# 			joint_state = self.env.reset()
# 			# joint_observation = joint_state
# 			#
# 			# belief = MultinomialDirichletBelief(self.env, joint_observation)
# 			sampled_joint_state = joint_state
# 			#
# 			# # create option pool
# 			# options, mu_policy = createOptions(self.env)
# 			# # options is a list of option object. Each option object has its own termination policy and pi_policy.
# 			# # pi_policy for option 0 can be called as	:	options[0].policy.weights
# 			# # options[0].policy is the object of SoftmaxActionPolicy()
# 			# # termination for option 0 can be called as	:	options[0].termination.weights
# 			#
# 			# terminations = [option.termination for option in options]
# 			# pi_policies = [option.policy for option in options]
# 			#
# 			# doc = DOC(self.env, options, mu_policy)
# 			#
# 			# d. Choose joint-option o based on softmax option-policy mu
# 			joint_option = self.doc.initializeOption(joint_state=joint_state)

# 			# # make the elected options unavailable
# 			# for option in joint_option:
# 			# 	self.options[option].available = False

# 			# joint action
# 			joint_action = self.doc.chooseAction()
# 			#
# 			# critic = IntraOptionQLearning(discount= params['train']['discount'],
# 			# 							  lr= params['train']['lr_critic'],
# 			# 							  terminations= terminations,
# 			# 							  weights= mu_policy.weights)
# 			#
# 			# action_critic = IntraOptionActionQLearning(discount= params['train']['discount'],
# 			# 										   lr = params['train']['lr_action_critic'],
# 			# 										   terminations=terminations,
# 			# 										   qbigomega=critic)
# 			#
# 			# agent_q = AgentQLearning(discount=params['train']['discount'],
# 			# 						 lr=params['train']['lr_agent_q'],
# 			# 						 options=options)
			
# 			self.critic.start(joint_state, joint_option)
# 			self.action_critic.start(joint_state,joint_option,joint_action)
# 			self.agent_q.start(joint_state, joint_option, joint_action)
			
# 			# termination_gradient = TerminationGradient(terminations, critic)
# 			# intra_option_policy_gradient = IntraOptionGradient(pi_policies)
			
# 			# done = False
# 			cum_reward = 0
# 			itr_reward = []

# 			next_joint_state = joint_state

# 			scale = 1.0

# 			for iteration in range(params['env']['episode_length']):
				
# 				# iv
# 				joint_action = self.doc.chooseAction()

# 				for agent in self.env.agents:
# 					agent.action = joint_action[agent.ID]
				
# 				# v
# 				reward, next_joint_state, done, _ = self.env.step(joint_action)
# 				#cum_reward += reward
				
# 				# vi - absorbed in broadcastBasedOnQ function of Broadcast class
				
# 				# vii - viii
# 				broadcasts = self.doc.toBroadcast(curr_true_joint_state=joint_state,
# 											 sampled_curr_joint_state=sampled_joint_state,
# 											 joint_option=joint_option,
# 											 done=done,
# 											 critic=self.critic,
# 											 reward=reward)

# 				step_reward = reward + np.sum([i*self.env.broadcast_penalty for i in broadcasts])
# 				cum_reward += scale*step_reward

				
# 				# ix
# 				next_joint_observation = self.env.get_observation(broadcasts)
# 				joint_observation = next_joint_observation
# 				sampled_joint_state = self.belief.sampleJointState(joint_observation) # iii

				
# 				# x - critic evaluation
# 				critic_feedback = self.doc.evaluateOption(critic=self.critic,
# 													 action_critic=self.action_critic,
# 													 joint_state=sampled_joint_state,
# 													 joint_option=joint_option,
# 													 joint_action=joint_action,
# 													 reward=step_reward,
# 													 done=done,
# 													 baseline=False)
				
# 				print('Iteration : ', iteration, 'step_reward : ', step_reward, 'Cumulative Reward : ', cum_reward, 'broadcasts : ', broadcasts, 'critic_feedback : ', critic_feedback)
				
# 				# xi A
# 				self.doc.improveOption(policy_obj=self.intra_option_policy_gradient,
# 								  termination_obj=self.termination_gradient,
# 								  joint_state=joint_state, #instead of sampled_joint_state
# 								  joint_option=joint_option,
# 								  joint_action=joint_action,
# 								  critic_feedback=critic_feedback
# 								   )
				
# 				# xi B
# 				joint_option = self.doc.chooseOptionOnTermination(self.options, joint_option, sampled_joint_state)
				
# 				joint_state = next_joint_state
				
				
# 				itr_reward.append(cum_reward)
# 				if not iteration%30:
# 					plotReward(itr_reward,'iterations','cumulative reward',self.expt_folder, 'iteration_reward.png')
				
# 				if done:
# 					break

# 				scale *= params['env']['discount']
					
# 			sum_of_rewards_per_episode.append(itr_reward[-1])
# 			plotReward(sum_of_rewards_per_episode, 'episodes', 'sum of rewards', self.expt_folder,
# 					   'single_episode_reward_per_episode.png')
	
		
	
		
# 	

from optionCritic.option import createOptions
from doc import DOC
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning, AgentQLearning
from distributed.belief import MultinomialDirichletBelief
from optionCritic.gradients import TerminationGradient, IntraOptionGradient
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from utils.viz import plotReward


class Trainer(object):
	def __init__(self, env, expt_folder):
		self.expt_folder = expt_folder
		self.env = env
		self.n_agents = params['env']['n_agents']
		
	def train(self):
		for _ in range(params['train']['n_runs']):
			# put the agents to the same initial joint state as long as the random seed set in params['train'][
			# 'seed'] in modelConfig remains unchanged
			joint_state = self.env.reset()
			joint_observation = joint_state


			alpha = 0.001 * np.ones(len(self.env.states_list))
			self.belief = MultinomialDirichletBelief(self.env,alpha)
			#self.belief = MultinomialDirichletBelief(self.env, joint_observation)
			# sampled_joint_state = joint_state
			
			# create option pool
			self.options, self.mu_policy = createOptions(self.env)
			# options is a list of option object. Each option object has its own termination policy and pi_policy.
			# pi_policy for option 0 can be called as	:	options[0].policy.weights
			# options[0].policy is the object of SoftmaxActionPolicy()
			# termination for option 0 can be called as	:	options[0].termination.weights
			
			terminations = [option.termination for option in self.options]
			pi_policies = [option.policy for option in self.options]
			
			self.doc = DOC(self.env, self.options, self.mu_policy)
			
			# # d. Choose joint-option o based on softmax option-policy mu
			# joint_option = self.doc.initializeOption(joint_state=joint_state)
			
			# # make the elected options unavailable
			# for option in joint_option:
			# 	self.options[option].available = False
			
			# joint action
			# joint_action = self.doc.chooseAction()
			
			self.critic = IntraOptionQLearning(discount=params['train']['discount'],
										  lr=params['train']['lr_critic'],
										  terminations=terminations,
										  weights=self.mu_policy.weights)
			
			self.action_critic = IntraOptionActionQLearning(discount=params['train']['discount'],
													   lr=params['train']['lr_action_critic'],
													   terminations=terminations,
													   qbigomega=self.critic)
			
			self.agent_q = AgentQLearning(discount=params['train']['discount'],
									 lr=params['train']['lr_agent_q'],
									 options=self.options)
			
			self.termination_gradient = TerminationGradient(terminations, self.critic)
			self.intra_option_policy_gradient = IntraOptionGradient(pi_policies)
			
			# for _ in range(params['train']['n_episodes']):
			self.trainEpisodes()
			

	def trainEpisodes(self):
		
		sum_of_rewards_per_episode = []
		for episode in range(params['train']['n_episodes']):
			#print('Episode : ', episode)
			# # put the agents to the same initial joint state as long as the random seed set in params['train'][
			# # 'seed'] in modelConfig remains unchanged
			joint_state = self.env.reset()
			# joint_observation = joint_state
			#
			# belief = MultinomialDirichletBelief(self.env, joint_observation)
			sampled_joint_state = joint_state
			old_feasible_states = joint_state
			#
			# # create option pool
			# options, mu_policy = createOptions(self.env)
			# # options is a list of option object. Each option object has its own termination policy and pi_policy.
			# # pi_policy for option 0 can be called as	:	options[0].policy.weights
			# # options[0].policy is the object of SoftmaxActionPolicy()
			# # termination for option 0 can be called as	:	options[0].termination.weights
			#
			# terminations = [option.termination for option in options]
			# pi_policies = [option.policy for option in options]
			#
			# doc = DOC(self.env, options, mu_policy)
			#
			# d. Choose joint-option o based on softmax option-policy mu
			joint_option = self.doc.initializeOption(joint_state=joint_state)

			# # make the elected options unavailable
			# for option in joint_option:
			# 	self.options[option].available = False

			# joint action
			joint_action = self.doc.chooseAction()
			#
			# critic = IntraOptionQLearning(discount= params['train']['discount'],
			# 							  lr= params['train']['lr_critic'],
			# 							  terminations= terminations,
			# 							  weights= mu_policy.weights)
			#
			# action_critic = IntraOptionActionQLearning(discount= params['train']['discount'],
			# 										   lr = params['train']['lr_action_critic'],
			# 										   terminations=terminations,
			# 										   qbigomega=critic)
			#
			# agent_q = AgentQLearning(discount=params['train']['discount'],
			# 						 lr=params['train']['lr_agent_q'],
			# 						 options=options)
			
			self.critic.start(joint_state, joint_option)
			self.action_critic.start(joint_state,joint_option,joint_action)
			self.agent_q.start(joint_state, joint_option, joint_action)
			
			# termination_gradient = TerminationGradient(terminations, critic)
			# intra_option_policy_gradient = IntraOptionGradient(pi_policies)
			
			# done = False
			cum_reward = 0
			itr_reward = []
			for iteration in range(params['env']['episode_length']):
				#print('Iteration : ', iteration, 'Cumulative Reward : ', cum_reward)


				# iv
				joint_action = self.doc.chooseAction()

				# Update agent's option and action

				for agent in self.env.agents:
					agent.option = joint_option[agent.ID]
					agent.action = joint_action[agent.ID]
				
				# v
				reward, next_joint_state, done, _ = self.env.step(joint_action)

				
				# vi - absorbed in broadcastBasedOnQ function of Broadcast class
				
				# vii - viii
				broadcasts = self.doc.toBroadcast(curr_true_joint_state=joint_state, #next_joint_state,
											 sampled_curr_joint_state=sampled_joint_state,
											 joint_option=joint_option,
											 done=done,
											 critic=self.critic,
											 reward=reward)
				
				step_reward = reward + np.sum([i*self.env.broadcast_penalty for i in broadcasts])

				cum_reward += step_reward

				# ix
				next_joint_observation = self.env.get_observation(broadcasts)
				joint_observation = next_joint_observation


				self.belief.update(joint_observation,old_feasible_states)
				sampled_joint_state = self.belief.sampleJointState()  #iii
				
				print('joint_obs', joint_observation, 'true_joint_state', joint_state, 'sampled_joint_state', sampled_joint_state)
				# x - critic evaluation
				critic_feedback = self.doc.evaluateOption(critic=self.critic,
													 action_critic=self.action_critic,
													 joint_state=sampled_joint_state, #joint_state,
													 joint_option=joint_option,
													 joint_action=joint_action,
													 reward=reward,
													 done=done,
													 baseline=False)
				
				print('Iteration : ', iteration, 'step_reward : ', step_reward, 'Cumulative Reward : ', cum_reward, 'broadcasts : ', broadcasts, 'critic_feedback : ', critic_feedback)
				
# 				
				# xi A
				self.doc.improveOption(policy_obj=self.intra_option_policy_gradient,
								  termination_obj=self.termination_gradient,
								  joint_state=sampled_joint_state,
								  joint_option=joint_option,
								  joint_action=joint_action,
								  critic_feedback=critic_feedback
								   )
				
				# xi B
				joint_option = self.doc.chooseOptionOnTermination(self.options, joint_option, sampled_joint_state)
				
				joint_state = next_joint_state
				old_feasible_states = belief.new_feasible_states(old_feasible_states,joint_observation)
				# joint_observation = next_joint_observation
				# sampled_joint_state = self.belief.sampleJointState(joint_observation) # iii
				
				itr_reward.append(cum_reward)
				if not iteration%30:
					plotReward(itr_reward,'iterations','cumulative reward',self.expt_folder, 'iteration_reward.png')
				
				if done:
					break
					
			sum_of_rewards_per_episode.append(itr_reward[-1])
			plotReward(sum_of_rewards_per_episode, 'episodes', 'sum of rewards', self.expt_folder,
					   'single_episode_reward_per_episode.png')
	
		
	
