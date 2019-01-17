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
		self.state = self.belief.sampleJointState()
		
		# policy over options
		self.mu_policy = mu_policy
		
	def chooseOption(self):
		#TODO: this module needs to be chnaged, since options are chosen as per spftmax policy
		# Choose joint-option o based on softmax option-policy
		
		#select agents randomly to pick options
		agent_order = [agent.ID for agent in self.env.agents]
		shuffle(agent_order)
		# print('agent order :', agent_order)
		
		joint_option = []
		#let agents select options from available option pool
		for agent in agent_order:
			option_mask = [not(option.available) for option in self.options]
			# print(option_mask)
			
			# pmf = [0, 0, 0.7, 0.1, 0.2]
			pmf = self.mu_policy.pmf(self.state[agent])
			pmf = np.ma.masked_array(pmf, option_mask)
			# print('pmf : ', pmf)
			
			# select option for agent
			# TODO : in order to sample option instead of choosing the best one, the masked pdf needs to be re-normalized
			selected_option_idx = np.argmax(pmf)
			self.env.agents[agent].option = self.options[selected_option_idx].optionID
			# print(selected_option_idx)
		
			#remove the selected option from available option pool by setting availability to False
			self.options[selected_option_idx].available = False
			
			joint_option.append(selected_option_idx)
			
		return tuple(joint_option)
			
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
			
		
	
			
	
		
			
		
		

		