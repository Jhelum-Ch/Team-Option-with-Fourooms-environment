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
		
	def chooseOption(self, joint_state):
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
			
	def chooseAction(self):
		joint_action = []
		for agent in self.env.agents:
			action = self.options[agent.option].policy.sample(agent.state)
			print('agent ID:', agent.ID, 'state:', agent.state, 'option ID:', agent.option, 'action:', action)
			joint_action.append(action)
			
		return tuple(joint_action)
	
	def evaluateOption(self, critic, action_critic, joint_state, joint_option, joint_action, baseline=False):
		
		reward, next_true_joint_state, done, _ = self.env.step(joint_action)
		
		broadcasts = Broadcast(self.env, next_true_joint_state, joint_state, joint_option,
							   done).broadcastBasedOnQ(critic, reward)
		
		# broadcasts = self.env.broadcast(reward, next_true_joint_state, self.s, self.o, terminations)
		joint_observation = self.env.get_observation(broadcasts)
		
		belief = MultinomialDirichletBelief(self.env, joint_observation)
		joint_state = belief.sampleJointState()
		
		# Critic update
		critic.update(joint_state, joint_option, reward, done)
		action_critic.update(joint_state, joint_option, joint_action, reward, done)
		
		critic_feedback = action_critic.getQvalue(joint_state, joint_option, joint_action)  # Q(s,o,a)
		
		if baseline:
			critic_feedback -= critic.value(joint_state, joint_option)
		return critic_feedback, done
	
	# run critic to calculate Q value
	
		# decide to broadcast, based on Q value
	
		# get joint observation y'
	
		# update belief, based on y' : update self.belief
	
		# sample joint state based on y' : update self.state
			
		
	
			
	
		
			
		
		

		