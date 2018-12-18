import torch
import torch.nn as nn

class MLP:
	def __init__(self, channels):
		super(MLP, self).__init__()
		

class VAIN:
	class FeatureEncoder(nn.Module):
		def __init__(self, num_agents):
			super(FeatureEncoder, self).__init__()
			self.fe_fc1 = nn.Linear(num_agents, 8)
			self.fe_fc2 = nn.Linear(8, 8)
			self.relu = nn.ReLU()
			
		def forward(self, input_feature):
			fe1 = self.relu(self.fe_fc1(input_feature))
			encoded_feature = self.relu(self.fe_fc2(fe1))
			return encoded_feature
		
	class
			
			
		
			