from optionCritic.option import createOptions
from doc import DOC
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning, AgentQLearning
import numpy as np


class Trainer(object):
	def __init__(self, env):
		self.env = env
		self.n_agents = params['env']['n_agents']
	
	def train(self):
		for _ in range(params['train']['n_epochs']):
			self.trainEpisode()
			
	# def putAgentsToGrid(self):
	# 	idx = 0
	# 	for agent in self.env.agents:
	# 		agent.state = params['env']['initial_joint_state'][idx]
	# 		idx += 1
	
	def trainEpisode(self):
		
		for episode in range(params['train']['n_episodes']):
			# put the agent to same initial joint state as long as uses the same random seed set in params['train'][
			# 'seed'] in modelConfig
			joint_state = self.env.reset()
			
			# # put agents at specific positions on the grid
			# self.putAgentsToGrid()
			
			# create option pool
			options, mu_policy = createOptions(self.env)
			# options is a list of option object. Each option object has its own termination policy and pi_policy.
			# pi_policy for option 0 can be called as	:	options[0].policy.weights
			# options[0].policy is the object of SoftmaxActionPolicy()
			# termination for option 0 can be called as	:	options[0].termination.weights
			
			terminations = [option.termination for option in options]
			
			doc = DOC(self.env, options, mu_policy)
			
			# d. Choose joint-option o based on softmax option-policy mu
			joint_option = doc.chooseOption(joint_state=joint_state)
			
			# joint action
			joint_action = doc.chooseAction()
			
			critic = IntraOptionQLearning(discount= params['train']['discount'],
										  lr= params['train']['lr_critic'],
										  terminations= terminations,
										  weights= mu_policy.weights)
			
			action_critic = IntraOptionActionQLearning(discount= params['train']['discount'],
													   lr = params['train']['lr_action_critic'],
													   terminations=terminations,
													   qbigomega=critic)
			
			x = [option.policy.weights for option in options]
			# print(x[0].shape)
			# import numpy as np; p = np.stack((x[:]), axis = 0)
			# print(p.shape)
			
			# agent_q = AgentQLearning(discount=params['train']['discount'],
			# 						 lr = params['train']['lr_agent_q'],
			# 						 weights=np.stack(([option.policy.weights for option in options][:]), axis = 0))
			
			agent_q = AgentQLearning(discount=params['train']['discount'],
									 lr=params['train']['lr_agent_q'],
									 options=options)
			
			
			critic.start(joint_state, joint_option)
			action_critic.start(joint_state,joint_option,joint_action)
			agent_q.start(joint_state, joint_option, joint_action)
			#print(agent_q.value(joint_state, joint_option, joint_action))
			
			# import pdb; pdb.set_trace()
			
			done = False
			for iteration in range(params['env']['episode_length']):
				
				
				# option evaluation
				critic_feedback, done = doc.evaluateOption(critic=critic,
														   action_critic=action_critic,
														   joint_state=joint_state,
														   joint_option=joint_option,
														   joint_action=joint_action,
														   baseline=False)
				
				if done:
					break
	
	
	
		
	