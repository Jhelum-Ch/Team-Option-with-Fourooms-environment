from modelConfig import params
from optionCritic.policies import SoftmaxPolicy
from optionCritic.termination import SigmoidTermination

class Option:
	def __init__(self, optionID, optionPolicy, optionTermination):
		super(Option, self).__init__()
		self.optionID = optionID
		self.policy = optionPolicy
		self.termination = optionTermination
		#self.broadcast = None
		self.available = True
		
def createOptions(env):
	options = []
	for i in range(params['agent']['n_options']):
		options.append(Option(i, SoftmaxPolicy(len(env.cell_list), len(env.agent_actions)), SigmoidTermination(len(env.cell_list))))
		
	# mu_policy is the policy over options
	mu_policy = SoftmaxPolicy(len(env.cell_list), params['agent']['n_options'])
		
	return options, mu_policy
		
		