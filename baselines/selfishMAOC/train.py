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
	def __init__(self, env, expt_folder, timestr):
		self.expt_folder = expt_folder
		self.env = env
		self.n_agents = params['env']['n_agents']
		self.writer = SummaryWriter(log_dir=expt_folder)
		self.timestr = timestr
		# self.output = []
		
	# def train(self):
	# 	for run in range(params['train']['n_runs']):
			# put the agents to the same initial joint state as long as the random seed set in params['train'][
			# 'seed'] in modelConfig remains unchanged
			
			# features = Tabular(self.env.observation_space.n)
			
			
			

		# 	self.trainEpisodes()

		
		# # Plots to report	
		# avg_len_epi = np.mean(self.steps_from_episode, axis=0)
		# avg_cum_rew = np.mean(self.cum_rew_from_episode, axis=0)
		# avg_crtic = np.mean(self.critic_from_episode, axis=0)
		# avg_dur = np.mean(self.avg_dur_from_episode, axis=0)

			

	def trainEpisodes(self):
		
		n_agent_states, n_actions = len(self.env.cell_list), params['agent']['n_actions']

		# create option pool
		#self.option_policies = [SoftmaxPolicy(self.rng, n_agent_states, n_actions, temp=params['policy']['temperature']) for _ in range(params['agent']['n_options']) ]

		self.options, self.mu_policies = createOptions(self.rng, self.env)
		# options is a list of option object. Each option object has its own termination policy and pi_policy.
		# pi_policy for option 0 can be called as	:	options[0].policy.weights
		# options[0].policy is the object of SoftmaxActionPolicy()
		# termination for option 0 can be called as	:	options[0].termination.weights
		
		#terminations = [option.termination for option in self.options]
		terminations = []
		for agent_idx in range(params['env']['n_agents']):
			terminations.append([option.termination for option in self.options[agent_idx]])
		
		self.doc = DOC(self.env, self.options, self.mu_policies)
		

		# list of opton-critics for all agents
		self.all_agent_critic = [IntraOptionQLearning(discount=params['env']['discount'],
									  lr=params['train']['lr_critic'],
									  terminations=terminations,
									  weights=self.mu_policies[i].weights) for i in range(params['env']['n_agents'])]


		# List of action-weights for all agents
		action_weights = [np.zeros((n_agent_states, params['agent']['n_options'], params['agent']['n_actions'])) for _ in range(params['env']['n_agents'])]
        #action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights, critic)

		#import pdb; pdb.set_trace()
		# list of Q(s,o,a) for all agents
		self.all_agent_action_critic = [IntraOptionActionQLearning(discount=params['env']['discount'],
												   lr=params['train']['lr_action_critic'],
												   terminations=terminations,
												   weights=action_weights[i],
												   qbigomega=self.all_agent_critic[i]) for i in range(params['env']['n_agents'])]
		
		
		self.termination_gradients = [TerminationGradient(self.options[i], self.all_agent_critic[i]) for i in range(params['env']['n_agents'])]
		self.intra_option_policy_gradients = [IntraOptionGradient(self.options[i]) for i in range(params['env']['n_agents'])]
			

		for episode in range(params['train']['n_episodes']):
			print('Episode : ', episode)
			
			# # put the agents to the same initial joint state as long as the random seed set in params['train'][
			# # 'f'] in modelConfig remains unchanged

			params['policy']['temperature'] = 1
			if params['policy']['temperature'] > 0.1:
				params['policy']['temperature'] -= 0.025
			else:
				params['policy']['temperature'] = 0.01
			joint_state = self.env.reset()
			
			
			# d. Choose joint-option o based on softmax option-policy mu
			joint_option = self.doc.initializeOption(joint_state=joint_state)

			# # make the elected options unavailable
			# for option in joint_option:
			# 	self.options[option].available = False

			# joint action
			joint_action = self.doc.chooseAction()


			# start all agents separately
			for i in range(params['env']['n_agents']):
				self.all_agent_critic[i].start(joint_state[i], joint_option[i])
				self.all_agent_action_critic[i].start(joint_state[i],joint_option[i],joint_action[i])
				# self.agent_q.start(joint_state, joint_option, joint_action)
			
			# termination_gradient = TerminationGradient(terminations, critic)
			# intra_option_policy_gradient = IntraOptionGradient(pi_policies)
			
			
			cum_rewards = [0. for _ in range(params['env']['n_agents'])]
			#all_agent_itr_rewards = [0.0 for _ in range(params['env']['n_agents'])]

			# options_episode = []
			# switch_agent = np.zeros(params['env']['n_agents'])

			c = 0.0
			
			for iteration in range(params['env']['episode_length']):


				if iteration % 50 == 0:
					print('Iteration : ', iteration, 'Cumulative Reward : ', np.mean(cum_rewards), 'Discovered Goals :', self.env.discovered_goals)


				if reward > 0 and params['policy']['temperature'] > 0.05:
					params['policy']['temperature'] -= 0.025


				#options_episode.append(joint_option)

				
				joint_action = self.doc.chooseAction()
				
				
				
				rewards, next_joint_state, done, _ = self.env.step(joint_action)
				rewards = [rewards[i] + c for i in range(params['env']['n_agents'])]
				
				
	
				# Get cumulative rewards for each agent
				
				cum_rewards = [rewards[i] + params['env']['discount'] * cum_rewards[i] for i in range(params['env']['n_agents'])]
			
				# x - critic evaluation
				all_agent_critic_feedback = self.doc.evaluateOption(self.all_agent_critic,
													 self.all_agent_action_critic,
													 next_joint_state,
													 joint_option,
													 joint_action,
													 rewards,
													 done,
													 baseline=False)

				
				self.doc.improveOption(self.intra_option_policy_gradients,
								  self.termination_gradients,
								  joint_state,
								  next_joint_state,
								  joint_option,
								  joint_action,
								  all_agent_critic_feedback
								   )
				

				next_joint_option, switch = self.doc.chooseOptionOnTermination(self.options, joint_option,
																			   joint_state) 
				#switches += switch
				
				if switch:
					c = 0.0001*params['train']['deliberation_cost']*switch
				else:
					c = 0

				for i in range(params['env']['n_agents']):
					if joint_option[i]!=next_joint_option[i]:
						switch_agent[i] += 1
				

				joint_option = next_joint_option
				
			
				joint_state = next_joint_state

				
				
			
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

			#TODO: Plot in average duration tensorboard




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




	
		
	