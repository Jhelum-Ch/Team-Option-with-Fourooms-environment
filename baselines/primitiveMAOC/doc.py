# from distributed.belief import Belief

from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning
from optionCritic.policies import SoftmaxOptionPolicy, SoftmaxActionPolicy

from distributed.belief import MultinomialDirichletBelief
from distributed.broadcast import Broadcast
from random import shuffle
import numpy as np
from modelConfig import params

class DOC:
	def __init__(self, env, options, mu_policy):
		'''
		:param states_list: all combination of joint states. This is an input from the environment
		:param lr_thea: list of learning rates for learning policy parameters (pi), for all the agents
		:param lr_phi: list of learning rates for learning termination functions (beta), for all the agents
		:param init_observation: list of joint observation of all the agents
		'''
		
		self.env = env
		self.options = options
		
		# set initial belief
		# initial_joint_observation = params['env']['initial_joint_state']
		# self.belief = MultinomialDirichletBelief(env, initial_joint_observation)
	
		# Sample a joint state s := vec(s_1,...,s_n) according to belief
		# self.joint_state = self.belief.sampleJointState()
		# since initial joint state is same as initial joint observation
		# self.joint_state = params['env']['initial_joint_state']
		
		# policy over options
		self.mu_policy = mu_policy
		
		# self.joint_option = self.chooseOption(joint_state=initial_joint_observation)  # since initial joint state is same as initial joint observation
		# self.joint_action = self.chooseAction()
		self.broadcast = Broadcast(self.env)
		
	def initializeOption(self, joint_state):
		# Choose joint-option o based on softmax option-policy
		joint_state = tuple(np.sort(joint_state))
		
		joint_option = self.mu_policy.sample(joint_state)
		
		for option in self.options:
			option.available = True
			
		for option in joint_option:
			self.options[option].available = False
		
		idx = 0
		for agent in self.env.agents:
			agent.option = joint_option[idx]
			idx += 1
			
		return joint_option
	
	def chooseOptionOnTermination(self, options, joint_option, joint_state):
		terminations = [1, 1, 1]
		
		for option in joint_option:
			options[option].available = True
		
		sampled_joint_option = self.mu_policy.sample(joint_state=tuple(np.sort(joint_state)))
		
		# make the options unavailable
		for option in sampled_joint_option:
			options[option].available = False
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
		for agent in self.env.agents:
			action = self.options[agent.option].policy.sample(agent.state)
			agent.action = action
			# print('agent ID:', agent.ID, 'state:', agent.state, 'option ID:', agent.option, 'action:', action)
			joint_action.append(action)
			
		return tuple(joint_action)
	
	
	def toBroadcast(self, curr_true_joint_state, prev_sampled_joint_state, prev_joint_obs, prev_true_joint_state, prev_joint_action, joint_option, done, critic, reward):
		return self.broadcast.broadcastBasedOnQ(critic, reward, curr_true_joint_state, prev_sampled_joint_state, prev_joint_obs, 
													prev_true_joint_state, prev_joint_action, joint_option,done)
	
	def evaluateOption(self, critic, action_critic, joint_state, joint_option, joint_action, reward,
					   done, baseline=False):
		# Critic update
		critic.update(joint_state, joint_option, reward, done)
		action_critic.update(joint_state, joint_option, joint_action, reward, done)
		
		critic_feedback = action_critic.getQvalue(joint_state, joint_option, joint_action)  # Q(s,o,a)
		
		if baseline:
			critic_feedback -= critic.value(joint_state, joint_option)
		return critic_feedback
	
	def improveOption(self, policy_obj, termination_obj, sampled_joint_state,
					  next_joint_state, estimated_next_joint_state, joint_option, joint_action, critic_feedback):
		# joint state refers to sampled state s_k and next_joint_state refers to s_k^'
		# update theta : policy improvement
		policy_obj.update(self.env.agents, sampled_joint_state, joint_option, joint_action, critic_feedback)
		
		#update phi : temriantion (beta) improvement
		termination_obj.update(next_joint_state, estimated_next_joint_state, joint_option)
	
	def improveOptionPrimitive(self, policy_obj, sampled_joint_state,
					  next_joint_state, estimated_next_joint_state, joint_option, joint_action, critic_feedback):
		# joint state refers to sampled state s_k and next_joint_state refers to s_k^'
		# update theta : policy improvement
		policy_obj.update(self.env.agents, sampled_joint_state, joint_option, joint_action, critic_feedback)
		
		# #update phi : temriantion (beta) improvement
		# termination_obj.update(next_joint_state, estimated_next_joint_state, joint_option)
	
			
	
		
			
		
		

		