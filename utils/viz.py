import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

def plotReward(cum_reward, xlabel, ylabel, location, title):

	plt.clf()
	
	plt.plot(range(len(cum_reward)), cum_reward, label='cumulative reward', color='g')
	
	plt.tight_layout()
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.legend()
	
	plt.savefig(os.path.join(location, title), bbox_inches="tight")
	
#TODO : Visualize belief

#TODO : visualize options