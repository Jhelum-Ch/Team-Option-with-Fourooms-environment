from utils.belief import Belief

class DOC:
	def __init__(self, env):
		'''
		:param states_list: all combination of joint states. This is an input from the environment
		:param lr_thea: list of learning rates for learning policy parameters (pi), for all the agents
		:param lr_phi: list of learning rates for learning termination functions (beta), for all the agents
		:param init_observation: list of joint observation of all the agents
		'''
		
		'''
		2. Start with initial common belief b_0
		'''
		# set initial observation
		self.b0 = Belief(env)
	
		'''
		3. Sample a joint state s := vec(s_1,...,s_n) according to b_0
		'''
		self.s = self.b0.sampleJointState()
		
		'''
		4. Choose joint-option o based on softmax option-policy
		'''
		

		