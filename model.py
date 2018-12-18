class DOC:
	def __init__(self, states_list, lr_thea, lr_phi, init_observation):
		'''
		:param states_list: all combination of joint states. This is an input from the environment
		:param lr_thea: list of learning rates for learning policy parameters (pi), for all the agents
		:param lr_phi: list of learning rates for learning termination functions (beta), for all the agents
		:param init_observation: list of joint observation of all the agents
		'''
		
		'''
		1. Input : learning rates alpha_theta and alpha_phi
		'''
		self.alpha_thea = lr_thea
		self.alpha_phi = lr_phi
		
		'''
		2. Start with initial observation and initial common belief b_0 based on y_0
		'''
		# set initial observation
		self.y_0 = init_observation
		
		#set initial common belief b_0 based on y_0
		# TODO : self.belief = same as initial observation. Vector of states of length = number of agents
	
		'''
		3. Sample a joint state s := vec(s_1,...,s_n) according to b_0
		'''
		self.s = self.sampleState()
		
	def sampleState(self):
		'''
		samples state based on current belief
		:return:
		'''
		pass

		