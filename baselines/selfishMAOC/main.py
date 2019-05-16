from selfish_fourroomsEnv import Selfish_FourroomsMA
from modelConfig import params, paths
from utils.unitTest import testCreateOption, testQ, testSoftmaxOptionPolicy #testActionSelection, testIntraOptionQLearning
from train import Trainer
import time
import os

def main():
	timestr = time.strftime("%Y%m%d-%H%M%S")
	base_folder = paths['output']['base_folder']
	expt_folder = base_folder + timestr
	if not os.path.exists(expt_folder):
		os.mkdir(expt_folder)
	
	print('Run : {}\n'.format(timestr))
	
	env = Selfish_FourroomsMA(n_agents=params['env']['n_agents'],
					  goal_reward = params['env']['goal_reward'],
					  collision_penalty = params['env']['collision_penalty'],
					  discount = params['env']['discount'])
	
	trainer = Trainer(env, expt_folder)
	trainer.train()
	
	
if __name__ == '__main__':
	main()