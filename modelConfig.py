# import numpy as np

# params = {
# 	'env'	:	{
# 		'discount'          : 0.9,
# 		'n_agents'			:	3,
# 		'goal_reward' 		:	 1,
# 		'broadcast_penalty'	:  -0.01,
# 		'collision_penalty' :  -0.01,
# 		'initial_joint_state' : (11,31,21)
# 	},
# 	'agent'	:	{
# 		'n_options'	:	5,
# 		'n_actions' :   4
# 	},
# 	'policy'	:	{
# 		'temperature'	: 0.5
# 	},
# 	'train': {
# 		'n_epochs'		: 	50,
# 		'n_episodes'	:	100,
# 		'n_steps' 		: 	100,
# 		'seed'			:	np.random.RandomState(1234)
# 	},
	
# 	'doc'	:	{
# 		'lr_theta'	: 0.01,
# 		'lr_phi'	: 0.01,
# 		'lr_Q'     : 0.01
# 	}
# }

import numpy as np

params = {
	'env'	:	{
		'n_agents'			:	3,
		'goal_reward' 		:	 1,
		'broadcast_penalty'	:  -0.01,
		'collision_penalty' :  -0.01,
		'episode_length'	: 	500,
		'no_broadcast_threshold' : 0.01,
		'selfishness_penalty' : -0.01,
		'discount' : 0.9
		#'initial_joint_state'	:	(11, 31, 21)
	},
	'agent'	:	{
		'n_options'	:	5,
		'n_actions'	:	4
	},
	'policy'	:	{
		'temperature'	: 0.5
	},
	'train': {
		'n_runs'		: 	1,
		'n_epochs'		: 	50,
		'n_episodes'	:	100,
		'n_steps' 		: 	500,
		'seed'			:	np.random.RandomState(1234),
		'discount'		:	0.1,
		'lr_critic'		: 	0.001,	#alpha_Q
		'lr_action_critic'	: 0.001,
		'lr_agent_q'		: 0.001,
		'lr_theta'			: 0.001,
		'lr_phi'			: 0.001
	}
}

paths = {
	'output'	:	{
		'base_folder'	:	'/home/ml/jchakr1/teamOptionResults/'
	}
}