from modelConfig import params
from optionCritic.option import Option
from optionCritic.policies import SoftmaxPolicy
from optionCritic.termination import SigmoidTermination
from optionCritic.option import createOptions
from random import shuffle
from doc import DOC

def testCreateOption(env):
	options, mu_policy = createOptions(env)
	
	print('policy over options :')
	for state in env.cell_list:
		print(mu_policy.pmf(state))
		
	for option in options:
		for state in env.cell_list:
			print(option.optionID, state, option.policy.pmf(state), option.termination.pmf(state))
			
def testSigmoidTermination(env):
	term = SigmoidTermination(env)
	state = 100
	print('pmf : ', term.pmf(state))
	print('sample : ', term.sample(state))
	print('grad : ', term.grad(state))	#don't know what it does!
	
def testOptionSelection(env):

	options, mu_policy = createOptions(env)
	doc = DOC(env, options, mu_policy)
	doc.chooseOption()

	for agent in env.agents:
		print(agent.name, agent.ID, agent.state, agent.option)
		
def testActionSelection(env):
	options, mu_policy = createOptions(env)
	doc = DOC(env, options, mu_policy)
	doc.chooseOption()
	joint_action = doc.chooseAction()
	doc.evaluateOption(joint_action)
	
	
	