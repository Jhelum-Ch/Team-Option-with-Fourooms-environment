from modelConfig import params
from optionCritic.policies import SoftmaxOptionPolicy, SoftmaxActionPolicy
from optionCritic.termination import SigmoidTermination
import itertools

class Option:
	def __init__(self, optionID, optionPolicy, optionTermination):
		super(Option, self).__init__()
		self.optionID = optionID
		self.policy = optionPolicy
		self.termination = optionTermination
		#self.broadcast = None
		self.available = True
		
def createOptions(env):
	joint_state_list = list(itertools.permutations(env.cell_list, env.n_agents))
	joint_option_list = list(itertools.permutations(range(params['agent']['n_options']), params['env']['n_agents']))
	joint_action_list = list(itertools.product(range(len(env.agent_actions)), repeat=params['env']['n_agents']))
	
	# mu_policy is the policy over options
	mu_weights = dict.fromkeys(joint_state_list, dict.fromkeys(joint_option_list, 0))	#TODO: fix this
	mu_policy = SoftmaxOptionPolicy(mu_weights)
	
	options = []
	for i in range(params['agent']['n_options']):
		options.append(Option(i, SoftmaxActionPolicy(len(env.cell_list), len(env.agent_actions)), SigmoidTermination(
			len(
			env.cell_list))))	#TODO create weight matrix for joint state and actions instead of individual
		
	return options, mu_policy
		
		