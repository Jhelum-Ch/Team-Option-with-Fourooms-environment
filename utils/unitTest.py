from modelConfig import params
from optionCritic.option import Option
from optionCritic.policies import SoftmaxPolicy
from optionCritic.termination import SigmoidTermination
from optionCritic.option import createOptions
from random import shuffle
from doc import DOC
from optionCritic.Qlearning import IntraOptionQLearning
import numpy as np
import itertools
import random
import operator


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
	
def testIntraOptionQLearning(env):
	options, mu_policy = createOptions(env)
	
	#agents need to choose options
	doc = DOC(env, options, mu_policy)
	doc.chooseOption()
	
	joint_options = list(itertools.permutations(range(5), 3))
	print('permutation : ', joint_options)
	
	weights = dict.fromkeys(env.states_list, dict.fromkeys(joint_options, random.randint(1, 10)))
	
	terminations = [option.termination for option in options]
	#print('terminations:', terminations)
	discount = 0.1
	lr = 0.1
	joint_state_from_belief = (1,2,3)
	joint_state = (5, 26, 64)
	joint_option = (4, 1, 2)
	
	q = IntraOptionQLearning(len(env.agents), discount, lr, terminations, weights)
	
	
	# test start()
	q.start(joint_state, joint_option)
	
	# test terminationProbOfAtLeastOneAgent()
	print('termination prob of at least one agent :', q.terminationProbOfAtLeastOneAgent(joint_state_from_belief,
																						 joint_option))
	
	# test getValue()
	print('get Q value :', q.getQvalue(joint_state, joint_option))
	print('get Q value with option=None:', q.getQvalue(joint_state))
	
	# test getAdvantage()
	print('advantage : ', q.getAdvantage(joint_state, joint_option))
	print('advantage with Option=None: ', q.getAdvantage(joint_state))
	
	# test update
	print('update :', q.update(joint_state, joint_option, reward=1, done=False))
	
def testIntraOptionActionQLearning(env):
	
	
	
		
	# test_terminationProbOfAtLeastOneAgent()
	# all_entries = q.getQvalue((5, 26, 64))
	# max_idx, max_val = max(all_entries.items(), key=operator.itemgetter(1))
	# print('max : ', max_val)
	# max_sub = {key:val - max_val for key, val in all_entries.items()}
	# print('max subtracted :', max_sub)
	

	# import operator as op
	# from functools import reduce
	#
	# def ncr(n, r):
	# 	r = min(r, n - r)
	# 	numer = reduce(op.mul, range(n, n - r, -1), 1)
	# 	denom = reduce(op.mul, range(1, r + 1), 1)
	# 	return numer / denom
	#
	# weights = np.ones((103*103*103, ncr(5,3)))
	# print(weights.shape)
	
	def testDict():
		# print(weights[(5, 26, 64)][(4, 1)])
		# print(weights[(5, 6, 64)])
		new_d = {a: max(b, key=b.get) for a, b in weights.items()}
		print(new_d)
		#max(weights.items(), key=operator.itemgetter(1))
		#print(weights[(5, 6, 64)].values())
	
	#testDict()

		
	
	
	