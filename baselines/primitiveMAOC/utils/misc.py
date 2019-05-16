import json
import torch
import os
import modelConfig

def saveModelandMetrics(modelObj):
	# Save Parameters
	with open(os.path.join(modelObj.expt_folder, 'parameters.json'), 'w') as fp:
		json.dump([modelConfig.params], fp)
	
	# Save the latest Trained Models #TODO : uncomment when algorithm is ready
	# torch.save(modelObj.model.state_dict(), os.path.join(modelObj.expt_folder, 'latest_model.pkl'))
	
	# TODO: Save metrics
	