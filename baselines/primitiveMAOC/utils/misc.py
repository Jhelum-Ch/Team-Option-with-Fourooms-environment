import json
import numpy as np
import os
import modelConfig
import pickle
from modelConfig import params

def saveModelandMetrics(modelObj):
	# Save Parameters
	with open(os.path.join(modelObj.expt_folder, 'parameters.json'), 'w') as fp:
		json.dump([modelConfig.params], fp)
	
	with open(os.path.join(modelObj.expt_folder, 'CriticWeights.pkl'), 'wb') as fp:
		pickle.dump(modelObj.critic.weights, fp)
		
	with open(os.path.join(modelObj.expt_folder, 'ActionCriticWeights.pkl'), 'wb') as fp:
		pickle.dump(modelObj.action_critic.weights, fp)

	with open(os.path.join(modelObj.expt_folder, 'MuPloicycWeights.pkl'), 'wb') as fp:
		pickle.dump(modelObj.mu_policy.weights, fp)
	
	for agent in range(params['env']['n_agents']):
		for idx, option in enumerate(modelObj.options[agent]):
			np.save(os.path.join(modelObj.expt_folder, 'PiPolicyWeights_Agent_%s_Option_%s.npy' % (agent, idx)), np.asarray(option.policy.weights))