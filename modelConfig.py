import numpy as np

params = {
	'env'	:	{
		'n_agents'			:	3,
		'goal_reward' 		:	 1,
		'broadcast_penalty'	:  -0.01,
		'collision_penalty' :  -0.01,
		'episode_length'	: 	500
	},
	'agent'	:	{
		'n_options'	:	5,
		'n_actions'	:	4
	},
	'policy'	:	{
		'temperature'	: 0.5
	},
	'train': {
		'n_epochs'		: 	50,
		'n_episodes'	:	100,
		'n_steps' 		: 	100,
		'seed'			:	np.random.RandomState(1234),
		'discount'		:	0.1,
		'lr_critic'		: 	0.01	#alpha_Q
	},
	
	'doc'	:	{
		'lr_theta'	: 0.01,
		'lr_phi'	: 0.01
	}
}