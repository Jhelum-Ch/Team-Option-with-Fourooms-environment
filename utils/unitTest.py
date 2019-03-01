from modelConfig import params
from optionCritic.option import Option
from optionCritic.policies import SoftmaxOptionPolicy, SoftmaxActionPolicy
from optionCritic.termination import SigmoidTermination
from optionCritic.option import createOptions
from random import shuffle
from doc import DOC
from distributed.belief import MultinomialDirichletBelief
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning
import numpy as np
import itertools
import random
import operator

from fourroomsEnv import FourroomsMA


def testCreateOption(env):
	options, mu_policy = createOptions(env)
	print(len(mu_policy.weights))
	joint_state = (5, 26, 64)
	mu_policy.weights[joint_state][(1,2,3)] = 10
	mu_policy.weights[joint_state][(0,3,4)] = 9
	print(mu_policy.weights[joint_state])
	print(mu_policy.sample(joint_state))
			
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
	
def testQ(env):
	options, mu_policy = createOptions(env)
	
	# agents need to choose options
	doc = DOC(env, options, mu_policy)
	doc.chooseOption()
	
	joint_options = list(itertools.permutations(range(params['agent']['n_options']), params['env']['n_agents']))
	joint_actions = list(itertools.product(range(4), repeat=params['env']['n_agents']))
	# print(len(joint_actions))
	terminations = [option.termination for option in options]
	print('terminations:', terminations)
	
	discount = 0.1
	lr = 0.1
	joint_state_from_belief = (1, 2, 3)
	joint_state = (5, 26, 64)
	joint_option = (4, 1, 2)
	joint_action = (0, 0, 2)
	
	def testIntraOptionQLearning():
		options, mu_policy = createOptions(env)
		
		#agents need to choose options
		doc = DOC(env, options, mu_policy)
		doc.chooseOption()
		
		# joint_options = list(itertools.permutations(range(5), 3))
		print('permutation : ', joint_options)
		
		weights = dict.fromkeys(env.states_list, dict.fromkeys(joint_options, 10))
		
		# option_list = list(range(5))
		# weights = dict.fromkeys(env.states_list, dict.fromkeys(option_list, 10))
		print('shape of Q :', len(weights), len(weights[joint_state]))
		
		# print('line 1 : ', weights[(5, 26, 64)])
		#
		# for opt in [1, 2, 3]:
		# 	print(weights[(5, 26, 64)][opt])
		#
		# print('sum of options 1, 2, 3 :', sum(weights[(5, 26, 64)][opt] for opt in (1, 2, 3)))
		
		# terminations = [option.termination for option in options]
		# print('terminations:', terminations)
		#
		# discount = 0.1
		# lr = 0.1
		# joint_state_from_belief = (1,2,3)
		# joint_state = (5, 26, 64)
		# joint_option = (4, 1, 2)
		
		q = IntraOptionQLearning(discount, lr, terminations, weights)
		
		
		# test start()
		q.start(joint_state, joint_option)
		
		# test terminationProbOfAtLeastOneAgent()
		print('termination prob of at least one agent :', q.terminationProbOfAtLeastOneAgent(joint_state_from_belief,
																							 joint_option))
		
		# test getValue()
		print('get Q value :', q.getQvalue(joint_state, joint_option))
		print('get Q value with option=None:', q.getQvalue(joint_state))
		print('get V value :', q.getQvalue(joint_state, None, joint_option))
		
		# test getAdvantage()
		print('advantage : ', q.getAdvantage(joint_state, joint_option))
		print('advantage with Option=None: ', q.getAdvantage(joint_state))
		
		# test update
		print('update :', q.update(joint_state, joint_option, reward=1, done=False))
		
		return q
		
	# q_omega = testIntraOptionQLearning()
		
	def testIntraOptionActionQLearning():
		
		weightsQ = dict.fromkeys(env.states_list, dict.fromkeys(joint_options, dict.fromkeys(joint_actions, 10)))
		aq = IntraOptionActionQLearning(discount, lr, terminations, weightsQ, q_omega)
		print('\nAction Q Value :', aq.getQvalue(joint_state, joint_option, joint_action))
		aq.start(joint_state, joint_option, joint_action)
		print('Action Q TD update :', aq.update(joint_state, joint_option, joint_action, reward=1, done=False))
		
	# testIntraOptionActionQLearning()
	
def testSoftmaxOptionPolicy(env):
	sl = env.states_list
	print('1. ', len(sl))
	sl = [tuple(np.sort(s)) for s in env.states_list]
	print('2. ',  len(sl))
	print(sl[0])
	sl = set(sl)
	print('3. ', len(sl))
	joint_options = list(itertools.permutations(range(params['agent']['n_options']), params['env']['n_agents']))
	joint_options = [tuple(np.sort(o)) for o in joint_options]
	joint_options = set(joint_options)
	# weights = dict.fromkeys(env.states_list, dict.fromkeys(joint_options, 10))
	weights = dict.fromkeys(sl, dict.fromkeys(joint_options, 10))
	# print(weights.keys())
	state = 30
	option = 3
	policy = SoftmaxOptionPolicy(weights)
	# policy.getQvalue(state)
	# print(policy.pmf(state))
	# policy.sample(joint_state=(20, 30, 40))
	print(policy.sample((30,)))

def testBelief(env):
	alpha = 0.001 * np.ones(len(env.states_list))
	belief = MultinomialDirichletBelief(env,alpha)

	#joint_observation = [(20,-1),(54,1),(102,1)]
	joint_observation = [(20,-1),(54,1),None]
	#joint_observation = [None,None,None]
	belief = belief.update(joint_observation)
	#belief.pmf()
	belief.sampleJointState()
		

env = FourroomsMA()

	
	
	# joint_state = (1, 2, 3)
	# joint_option = (0, 1, 3)
	# # mat_state = []
	# # mat_option = []
	# # for k1, v1 in weights.items():
	# # 	if k1[0] == joint_state[0]:
	# # 		mat_option = []
	# # 		# print(k1)
	# # 		# mat.append(weights[k1])
	# # 		# sum0 = 0
	# # 		for k2, v2 in v1.items():
	# # 			if(k2[0] == 0):
	# # 				mat_option.append(weights[k1][k2])
	# # 				# sum0 += v2
	# # 		#
	# # 		# print(sum0)
	# # 	if len(mat_option) >0:
	# # 		mat_state.append(mat_option)
	# #
	# # print(np.array(mat_state).shape)
	#
	# option_keys = weights[(0, 1, 2)].keys()
	#
	# for agent_idx in range(3):
	# 	agent_state_keys = [s for s in weights.keys() if s[agent_idx] == joint_state[agent_idx]]
	# 	print(len(agent_state_keys))
	#
	# 	for option_idx in range(5):
	# 		option_key = [o for o in option_keys if o[option_idx] == joint_option[agent_idx]]
	# 		print(option_key)
	#
	# 		for option in option_key:
	# 			summ = 0
	# 			for state in agent_state_keys:
	# 				summ  += weights[state][option]
	#
	# 		print('agent_idx :', agent_idx, 'optionQ :', summ)
	
	





	
	
		
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
	
	# def testDict():
	# 	# print(weights[(5, 26, 64)][(4, 1)])
	# 	# print(weights[(5, 6, 64)])
	# 	new_d = {a: max(b, key=b.get) for a, b in weights.items()}
	# 	print(new_d)
	# 	#max(weights.items(), key=operator.itemgetter(1))
	# 	#print(weights[(5, 6, 64)].values())
	#
	# #testDict()

		
	
	
	
