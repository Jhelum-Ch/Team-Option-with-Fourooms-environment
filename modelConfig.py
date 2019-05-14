import numpy as np

params = {
	'env'	:	{
		'n_agents'			:	3,
		'goal_reward' 		:	 1,
		'broadcast_penalty'	:  -0.02,
		'collision_penalty' :  -0.01,
		'episode_length'	: 	1000,
		'no_broadcast_threshold'	: 0.01,	#TODO : tune
		'selfishness_penalty'	: -0.01,	#TODO : tune
		'discount'				: 0.99
		#'initial_joint_state'	:	(11, 31, 21)
	},
	'agent'	:	{
		'n_options'	:	5,
		'n_actions'	:	4
	},
	'policy'	:	{
		'temperature'	: 0.6
	},
	'train': {
		'n_runs'		: 	1,
		# 'n_epochs'		: 	50,
		'n_episodes'	:	200,
		'n_steps' 		: 	1000,
		'seed'			:	1234,
		# 'discount'		:	0.1,
		'lr_critic'		: 	0.1,	#alpha_Q
		'lr_action_critic'	: 0.1,
		'lr_agent_q'		: 0.1,
		'lr_theta'			: 0.1,
		'lr_phi'			: 0.1,
		'deliberation_cost' : 0.1
	}
}

seed = np.random.RandomState(params['train']['seed'])

paths = {
	'output'	:	{
		'base_folder'	:	'/home/ml/sbasu11/Documents/MultiAgent/experiments/'
	}
}