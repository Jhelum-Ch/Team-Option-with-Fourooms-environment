from optionCritic.belief import Belief
from random import shuffle
import numpy as np

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
		self.belief = Belief(env)
	
		# Sample a joint state s := vec(s_1,...,s_n) according to belief
		self.joint_state = self.belief.sampleJointState()
		
		# policy over options
		self.mu_policy = mu_policy
		
	def chooseOption(self):
		# Choose joint-option o based on softmax option-policy
		
		joint_option = self.mu_policy.sample(self.joint_state)
		
		for option in self.options:
			option.available = True
			
		for option in joint_option:
			self.options[option].available = False
			
		return joint_option
			
	def chooseAction(self):
		joint_action = []
		for agent in self.env.agents:
			action = self.options[agent.option].policy.sample(agent.state)
			print('agent ID:', agent.ID, 'state:', agent.state, 'option ID:', agent.option, 'action:', action)
			joint_action.append(action)
			
		return tuple(joint_action)
	
	def evaluateOption(self, joint_state, joint_action):
		reward, done, _ = self.env.step(joint_action)
		print(reward, done)
		
		# run critic to calculate Q value
	
		# decide to broadcast, based on Q value
	
		# get joint observation y'
	
		# update belief, based on y' : update self.belief
	
		# sample joint state based on y' : update self.state
			
		
	
			
	
		
			
		
		

		