from fourroomsEnv import FourroomsMA
from modelConfig import params
from utils.unitTest import testCreateOption, testQ, testSoftmaxOptionPolicy #testActionSelection, testIntraOptionQLearning
from train import Trainer

def main():
	env = FourroomsMA(n_agents=params['env']['n_agents'],
					  goal_reward = params['env']['goal_reward'],
					  broadcast_penalty =params['env']['broadcast_penalty'],
					  collision_penalty = params['env']['collision_penalty'])
	
	trainer = Trainer(env)
	trainer.trainEpisode()
	
	# testCreateOption(env)
	# testOptionSelection(env)
	#testActionSelection(env)
	# testIntraOptionQLearning(env)
	#print(env.states_list)
	#print(len(env.states_list))
	
	# testQ(env)
	# testSoftmaxOptionPolicy(env)
	# print(env.currstate)
	
	
	
if __name__ == '__main__':
	main()