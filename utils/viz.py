import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
from modelConfig import params

def plotReward(cum_reward, xlabel, ylabel, location, title):

	plt.clf()
	
	plt.plot(range(len(cum_reward)), cum_reward, label='cumulative reward', color='g')
	
	plt.tight_layout()
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.legend()
	
	plt.savefig(os.path.join(location, title), bbox_inches="tight")

def calcErrorInBelief(env, true_joint_state, sampled_joint_state):
	error = 0
	for true_state, sampled_state in zip(true_joint_state, sampled_joint_state):
		# print('true state :', true_state, 'tocell :', env.tocellcoord[true_state])
		# print('sampled state :', sampled_state, 'tocell :', env.tocellcoord[sampled_state])
		error += np.linalg.norm([i - j for (i, j) in zip(env.tocellcoord[true_state], env.tocellcoord[
			sampled_state])], 2)
	
	return error

def calcCriticValue(nested_dict):
	keys = list(nested_dict.keys())
	Q = []
	for key in keys:
		Q.append(max(nested_dict[key].values()))
	return np.linalg.norm(Q) * 1.0 / len(Q)

# def calcActionCriticValue(nested_dict):
# 	state_keys = list(nested_dict.keys())
# 	Q = []
# 	for state_key in state_keys:
# 		option_keys = list(nested_dict[state_key].keys())
# 		for option in range(params['agent']['n_options']):
# 			Q.append(sum(nested_dict[o_key][a_key].values())
# 	return total_Q

def calcActionCriticValue(nested_dict):
	option_keys = list(nested_dict.keys())
	total_Q = 0
	for o_key in option_keys:
		action_keys = list(nested_dict[o_key].keys())
		for a_key in action_keys:
			total_Q += sum(nested_dict[o_key][a_key].values())
	return total_Q
	
def calcAgentActionValue(options):
	action_values = []
	for option in options:
		action_values.append(np.linalg.norm(np.max(option.policy.weights, axis= 1)))
	return action_values

#TODO : visualize options

def calcOptionValue(option_Q):
	v = np.max(option_Q, axis=1)
	return np.linalg.norm(v) * 1.0 / v.shape[0]