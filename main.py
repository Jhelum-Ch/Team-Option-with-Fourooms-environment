from fourroomsEnv import FourroomsMA
from modelConfig import params, paths
# from utils.unitTest import testCreateOption, testQ, testSoftmaxOptionPolicy #testActionSelection, testIntraOptionQLearning
from train import Trainer
import matplotlib.pyplot as plt
import time
import os

def main():
	timestr = time.strftime("%Y%m%d-%H%M%S")
	base_folder = paths['output']['base_folder']
	expt_folder = base_folder + timestr
	if not os.path.exists(expt_folder):
		os.mkdir(expt_folder)
	
	print('Run : {}\n'.format(timestr))
	
	env = FourroomsMA(n_agents=params['env']['n_agents'],
					  goal_reward = params['env']['goal_reward'],
					  broadcast_penalty =params['env']['broadcast_penalty'],
					  collision_penalty = params['env']['collision_penalty'])
	
	trainer = Trainer(env, expt_folder)
	all_run_steps, all_run_cum_rew = trainer.train(pbar = None)
	mu1 = all_run_steps.mean(axis=0)
	sigma1 = all_run_steps.std(axis=0)
	mu2 = all_run_cum_rew.mean(axis=0)
	sigma2 = all_run_cum_rew.std(axis=0)

	episodes = all_run_steps.shape[1]

	# Plot
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)

	ax1.plot(episodes, mu1, color='black')
	ax1.fill_between(episodes, mu1 + sigma1, mu1 - sigma1, facecolor='grey', alpha=0.5)
	ax1.set_xlabel('number of episodes')
	ax1.set_ylabel('Average episode lengths')



	ax2.plot(episodes, mu2, color='black')
	ax1.fill_between(episodes, mu2 + sigma2, mu2 - sigma2, facecolor='grey', alpha=0.5)
	ax2.set_xlabel('number of episodes')
	ax2.set_ylabel('Average cumulative rewards')

	plt.show()

	fig.savefig("plots.pdf", bbox_inches='tight')







	
	
if __name__ == '__main__':
	main()