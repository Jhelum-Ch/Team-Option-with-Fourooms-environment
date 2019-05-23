# from distributed.belief import Belief

from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning
# from optionCritic.policies import SoftmaxOptionPolicy, SoftmaxActionPolicy
from optionCritic.policies import SoftmaxPolicy

#from distributed.belief import MultinomialDirichletBelief
#from distributed.broadcast import Broadcast
from random import shuffle
import numpy as np
from modelConfig import params

class DOC:
	def __init__(self, env, options, mu_policies):
		'''
		:param states_list: all combination of joint states. This is an input from the environment
		:param lr_thea: list of learning rates for learning policy parameters (pi), for all the agents
		:param lr_phi: list of learning rates for learning termination functions (beta), for all the agents
		:param init_observation: list of joint observation of all the agents
		'''
		
		self.env = env
		self.options = options # this is pool of options
		
		# set initial belief
		# initial_joint_observation = params['env']['initial_joint_state']
		# self.belief = MultinomialDirichletBelief(env, initial_joint_observation)
	
		# Sample a joint state s := vec(s_1,...,s_n) according to belief
		# self.joint_state = self.belief.sampleJointState()
		# since initial joint state is same as initial joint observation
		# self.joint_state = params['env']['initial_joint_state']
		
		# policy over options
		self.mu_policies = mu_policies
		
		# self.joint_option = self.chooseOption(joint_state=initial_joint_observation)  # since initial joint state is same as initial joint observation
		# self.joint_action = self.chooseAction()
		#self.broadcast = Broadcast(self.env)
		
	def initializeOption(self, joint_state):
		# Choose joint-option o based on softmax option-policy
		joint_state = tuple(np.sort(joint_state))		# TODO: Double-check if sorting makes sense here.
		joint_option = np.zeros(len(joint_state), dtype=int)

		for i, agent_state in enumerate(joint_state):
			# print('weights',self.mu_policies[0].weights.shape[0])
			joint_option[i] = self.mu_policies[i].sample(agent_state)	# TODO: options sampled with replacement here, but availability is set in code below.
																		# TODO: (continued) This seems inconsistent.
			
		#print("Joint option: ", joint_option)

		# for option in self.options:
		# 	option.available = True
			
		# for option in joint_option:
		# 	self.options[option].available = False
		
		idx = 0
		for agent in self.env.agents:
			agent.option = joint_option[idx]
			idx += 1
			
		return joint_option
	
	def chooseOptionOnTermination(self, options, joint_option, joint_state):
		terminations = [1, 1, 1]
		
		for option in joint_option:
			options[option].available = True
		
		sampled_joint_option = [self.mu_policies[i].sample(joint_state[i]) for i in range(len(joint_state))]
		
		# make the options unavailable
		# for option in sampled_joint_option:
		# 	options[option].available = False
		# return the joint options

		return sampled_joint_option, np.sum(terminations)

	
	# def chooseOptionOnTermination(self, options, joint_option, joint_state):
	# 	terminations = []
	# 	for agentID in range(len(joint_state)):
	# 		state = joint_state[agentID]
	# 		option = joint_option[agentID]
	#
	# 		# for each agent sample from termination. This gives a boolean value representing whether the option terminates
	# 		terminate = options[option].termination.sample(state)
	#
	# 		# # make the options that terminate available
	# 		# if terminate:
	# 		# 	options[option].available = True
	# 		terminations.append(terminate)
	#
	# 	# print('termination : ', terminations)
	# 	# print('previous options :', joint_option)
	#
	# 	# available_options = [option.optionID for option in options if option.available]
	#
	# 	# if none of the options terminated, return the existing joint option
	# 	if not np.sum(terminations):
	# 		return joint_option, np.sum(terminations)
	#
	# 	# # if at least one of the agent is terminating, sample a joint option from mu_policy that conforms with the
	# 	# # non-terminating options
	# 	# #To account for options that did not terminate and the available options
	# 	# feasible_joint_option = [None, None, None]
	# 	# for idx, term in enumerate(terminations):
	# 	# 	if not term:
	# 	# 		feasible_joint_option[idx] = joint_option[idx]
	# 	#
	# 	# selected = []
	# 	# candidates = self.mu_policy.weights[tuple(np.sort(joint_state))]
	# 	# print('candidates:',candidates)
	# 	# # print('from mu policy :', self.mu_policy.weights[tuple(np.sort(joint_state))])
	# 	# for candidate in candidates:
	# 	# 	condition = True
	# 	# 	for i, fe in enumerate(feasible_joint_option):
	# 	# 		if not fe is None:
	# 	# 			if not fe == candidate[i]:
	# 	# 				condition = False
	# 	# 				break
	# 	# 	if condition:
	# 	# 		selected.append(candidate)
	# 	#
	# 	# print(selected)
	#
	# 	# terminate all the options and make them available
	# 	for option in joint_option:
	# 		options[option].available = True
	#
	# 	sampled_joint_option  = self.mu_policy.sample(joint_state = tuple(np.sort(joint_state)))
	#
	# 	# make the options unavailable
	# 	for option in sampled_joint_option:
	# 		options[option].available = False
	# 	# return the joint options
	# 	return sampled_joint_option, np.sum(terminations)
	
			
	def chooseAction(self):
		joint_action = []
		for idx, agent in self.env.agents:
			action = self.options[idx][agent.option].policy.sample(agent.state)
			agent.action = action
			# print('agent ID:', agent.ID, 'state:', agent.state, 'option ID:', agent.option, 'action:', action)
			joint_action.append(action)
			
		return tuple(joint_action)
	
	
	# def toBroadcast(self, curr_true_joint_state, prev_sampled_joint_state, prev_joint_obs, prev_true_joint_state, prev_joint_action, joint_option, done, critic, reward):
	# 	return self.broadcast.broadcastBasedOnQ(critic, reward, curr_true_joint_state, prev_sampled_joint_state, prev_joint_obs, 
	# 												prev_true_joint_state, prev_joint_action, joint_option,done)
	
	def evaluateOption(self, all_agent_critic, all_agent_action_critic, joint_state, joint_option, joint_action, rewards,
					   done, baseline=False):

		all_agent_critic_feedback = [0. for i in range(params['env']['n_agents'])] 

		for agent in range(params['env']['n_agents']):
			# Critic update
			all_agent_critic[agent].update(joint_state[agent], joint_option[agent], rewards[agent], done)
			all_agent_action_critic[agent].update(joint_state[agent], joint_option[agent], joint_action[agent], rewards[agent], done)
			
			agent_critic_feedback = all_agent_action_critic[agent].value(joint_state[agent], joint_option[agent], joint_action[agent])  # Q(s,o,a)
			all_agent_critic_feedback[agent] = agent_critic_feedback
			
			if baseline:
				agent_critic_feedback -= all_agent_critic[agent].value(joint_state[agent], joint_option[agent])
				all_agent_critic_feedback[agent] = agent_critic_feedback
			return all_agent_critic_feedback
	
	def improveOption(self, all_agent_policy_obj, all_agent_termination_obj, joint_state,
					  next_joint_state, joint_option, joint_action, all_agent_critic):
		# joint state refers to sampled state s_k and next_joint_state refers to s_k^'
		# update theta : policy improvement
		for agent in range(params['env']['n_agents']):
			all_agent_policy_obj[agent].update(joint_state[agent], joint_option[agent], joint_action[agent], all_agent_critic[agent])
			
			#update phi : temriantion (beta) improvement
			all_agent_termination_obj[agent].update(next_joint_state[agent], joint_option[agent])
	
			
	
		
			
		
		

		