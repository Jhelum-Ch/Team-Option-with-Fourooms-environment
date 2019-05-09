import json
import numpy as np
import os
import modelConfig
import dill
import pickle

def saveModelandMetrics(modelObj):
	# Save Parameters
	with open(os.path.join(modelObj.expt_folder, 'parameters.json'), 'w') as fp:
		json.dump([modelConfig.params], fp)
		
	with open(os.path.join(modelObj.expt_folder, 'InterOptionPloicycWeights.pkl'), 'wb') as fp:
		pickle.dump(modelObj.mu_policy.weights, fp)
	
	with open(os.path.join(modelObj.expt_folder, 'CriticWeights.pkl'), 'wb') as fp:
		pickle.dump(modelObj.critic.weights, fp)
		
	with open(os.path.join(modelObj.expt_folder, 'ActionCriticWeights.pkl'), 'wb') as fp:
		pickle.dump(modelObj.action_critic.weights, fp)
		
	with open(os.path.join(modelObj.expt_folder, 'WeightInterOptions.pkl'), 'wb') as fp:
		pickle.dump(modelObj.mu_policy.weights, fp)
		
	# with open(os.path.join(modelObj.expt_folder, 'model_object.pkl'), 'wb') as fp:
	# 	dill.dump(modelObj, fp)
	
	# np.save(os.path.join(modelObj.expt_folder, 'CriticWeight.npy'), np.asarray(modelObj.critic.weights))
	
	# np.save(os.path.join(modelObj.expt_folder, 'WeightInterOptions.npy'), np.asarray(modelObj.mu_policy.weights))
	
	for idx, option in enumerate(modelObj.options):
		np.save(os.path.join(modelObj.expt_folder, 'Weight_Options_%s.npy' % idx), np.asarray(option.policy.weights))
		
	
	
	# Save the latest Trained Models #TODO : uncomment when algorithm is ready
	# torch.save(modelObj.model.state_dict(), os.path.join(modelObj.expt_folder, 'latest_model.pkl'))
	
	# TODO: Save metrics
	