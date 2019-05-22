import numpy as np

params = {
	'env'	:	{
		'n_agents'			:	2, #2,
		'goal_reward' 		:	 2., #1,
		'broadcast_penalty'	:  -0.02,
		'collision_penalty' :  0.0, #-0.01,
		'episode_length'	: 	1500,
		'no_broadcast_threshold'	: 0.01,	#TODO : tune
		'selfishness_penalty'	: -0.01,	#TODO : tune
		'discount'				: 0.99
		#'initial_joint_state'	:	(11, 31, 21)
	},
	'agent'	:	{
		'n_options'	:	3,
		'n_actions'	:	4
	},
	'policy'	:	{
		'temperature'	: 1.0,
		'epsilon' : 0.05
	},
	'train': {
		'n_runs'		: 	1,
		# 'n_epochs'		: 	50,
		'n_episodes'	:	200,
		# 'n_steps' 		: 	1000,
		'seed'			:	42,
		# 'discount'		:	0.1,
		'lr_critic'		: 	0.5,	#alpha_Q
		'lr_action_critic'	: 0.5,
		'lr_agent_q'		: 0.1,
		'lr_theta'			: 0.1,
		'lr_phi'			: 0.1,
		'deliberation_cost' : 0.1
	}
}

seed = np.random.RandomState(params['train']['seed'])

paths = {
	'output'	:	{

		#'base_folder'	:	'/private/home/sumanab/multiagent/experiments/' #'/home/ml/sbasu11/Documents/MultiAgent/experiments/'
		'base_folder'	:	'/home/ml/jchakr1/teamOptionResults/'
	}
}