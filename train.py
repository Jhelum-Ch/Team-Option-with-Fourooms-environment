
from modelConfig import params
from fourroomsEnv import FourroomsMA
from optionCritic.belief import Belief
from optionCritic.option import createOptions
from doc import DOC
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning

class Trainer:
	def __init__(self, env):
		self.env = env
		self.n_agents = params['env']['n_agents']
		
		# create option pool
		self.options, self.mu_policy = createOptions(self.env)
		self.terminations = [option.termination for option in self.options]
	
	def train(self):
		for _ in range(params['train']['n_epochs']):
			self.trainEpisode()
	
	def trainEpisode(self):
		
		for _ in range(params['train']['n_episodes']):
			# a. Start with an initial common belief b0
			# c. Sample a joint-states:= (s1,...,sn) according to b0
			doc = DOC(self.env, self.options, self.mu_policy)
			
			# d. Choose joint-option o based on softmax option-policy mu
			joint_option = doc.chooseOption()
			
			critic = IntraOptionQLearning(discount= params['train']['discount'],
										  lr= params['train']['lr_critic'],
										  terminations= self.terminations,
										  self.mu_policy.weights)
			
			
			done = False
			for episode in range(params['env']['episode_length']):
				# create/reset critic
				if done:
					break
				done = self.trainStep(doc)
			
	def trainStep(self, doc):
		joint_state = doc.state
		joint_action = doc.chooseAction()
		
		doc.evaluateOption(joint_state, joint_action)
		
		return done
	
			
			
			
	
	
	def trainStep(self):
		#1. Sample s_t ~ b_t
		pass
	
	
		
	