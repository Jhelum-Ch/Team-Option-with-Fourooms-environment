import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np

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


#TODO : visualize options