import numpy as np

params = {
	'env'	:	{
		'n_agents'			:	2, #2,
		'goal_reward' 		:	5, #1,
		'broadcast_penalty'	: 	-0.02, #-0.02,
		'collision_penalty' :  0.0, #-0.01,
		'episode_length'	: 	800,
		'no_broadcast_threshold'	: 0.0, #0.01,
		'selfishness_penalty'	: 0.0, #-0.01,
		'discount'				: 0.99, #0.99
	},
	'agent'	:	{
		'n_options'	:	3,
		'n_actions'	:	4
	},
	'policy'	:	{
		'temperature'	: 1.0,
		'epsilon' : 0.1 # for epsilon greedy
	},
	'train': {
		# 'n_runs'		: 	10,
		'n_episodes'	:	20,
		'seed'			:	np.random.randint(100), #42,
		'lr_critic'		: 	0.1,	#for Q(s,o) and Q(s,o,a) alpha_Q
		# 'lr_action_critic'	: 0.5,
		'lr_agent_q'		: 0.005,
		'lr_theta'			: 0.01,	# policy LR
		'lr_phi'			: 0.01,	# termination LR
		'deliberation_cost' : 0	#0.1
	}
}

seed = np.random.RandomState(params['train']['seed'])

paths = {
	'output'	:	{

		# 'base_folder'	:	'/private/home/sumanab/multiagent/experiments/',
		'base_folder'	:	'/private/home/sumanab/checkpoint/multiagent/experiments4/',
		#'base_folder'	:	'/home/ml/sbasu11/Documents/MultiAgent/experiments/',
		'graphs_folder'	:	'/private/home/sumanab/checkpoint/multiagent/graphs4/'
	}
}