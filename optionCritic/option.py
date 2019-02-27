# from modelConfig import params
# from optionCritic.policies import SoftmaxOptionPolicy, SoftmaxActionPolicy
# from optionCritic.termination import SigmoidTermination
# import itertools
# import numpy as np

# class Option:
# 	def __init__(self, optionID, optionPolicy, optionTermination):
# 		super(Option, self).__init__()
# 		self.optionID = optionID
# 		self.policy = optionPolicy
# 		self.termination = optionTermination
# 		#self.broadcast = None
# 		self.available = True
		
# def createOptions(env):
# 	joint_state_list = set([tuple(np.sort(s)) for s in env.states_list])
# 	joint_option_list = list(itertools.permutations(range(params['agent']['n_options']), params['env']['n_agents']))
# 	joint_action_list = list(itertools.product(range(len(env.agent_actions)), repeat=params['env']['n_agents']))
	
# 	# mu_policy is the policy over options
# 	mu_weights = dict.fromkeys(joint_state_list, dict.fromkeys(joint_option_list, 0))
# 	mu_policy = SoftmaxOptionPolicy(mu_weights)
	
# 	options = []
# 	for i in range(params['agent']['n_options']):
# 		options.append(Option(i, SoftmaxActionPolicy(len(env.cell_list), len(env.agent_actions)), SigmoidTermination(
# 			len(env.cell_list))))
		
# 	return options, mu_policy
		
# 		

from modelConfig import params
from optionCritic.policies import SoftmaxOptionPolicy, SoftmaxActionPolicy
from optionCritic.termination import SigmoidTermination
import itertools
import numpy as np

class Option:
	def __init__(self, optionID, optionPolicy, optionTermination):
		super(Option, self).__init__()
		self.optionID = optionID
		self.policy = optionPolicy
		self.termination = optionTermination
		#self.broadcast = None
		self.available = True
		
def createOptions(env):
	'''
	:param env:
	:return:
		options : list of option objects
		mu_policy = Softmax option policy object
	'''
	joint_state_list = set([tuple(np.sort(s)) for s in env.states_list])
	all_joint_options =  list(itertools.permutations(range(params['agent']['n_options']), params['env']['n_agents']))
	joint_option_list = set([tuple(np.sort(jo)) for jo in all_joint_options])
	
	# mu_policy is the policy over options
	mu_weights = dict.fromkeys(joint_state_list, dict.fromkeys(joint_option_list, 0))
	mu_policy = SoftmaxOptionPolicy(mu_weights)
	
	options = []
	for i in range(params['agent']['n_options']):
		options.append(Option(i, SoftmaxActionPolicy(len(env.cell_list), len(env.agent_actions)), SigmoidTermination(
			len(env.cell_list))))
		
	return options, mu_policy
		
		
