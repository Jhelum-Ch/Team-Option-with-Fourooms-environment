import torch
import numpy as np
from numpy import random


class FeatureExtractNet(torch.nn.Module):

	# Input: states
	# Output: features


	def __init__(self, D_in, H, D_out):
	
		super(FeatureExtractNet, self).__init__()
		self.fc1 = torch.nn.Linear(D_in, H)
		self.fc2 = torch.nn.Linear(H, D_out)
		self.bn1 = torch.nn.BatchNorm1d(H)
		self.bn2 = torch.nn.BatchNorm1d(D_out)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		
		h_relu = self.relu(self.bn1(self.fc1(x).clamp(min=0)))
		feature_pred = self.relu(self.bn2(self.fc2(h_relu)))
		return feature_pred

class EsNet(torch.nn.module):

	# Input: features
	# Output: e^{s,i} for agent i


	def __init__(self, D_in, H, D_out):
	
		super(EsNet, self).__init__()
		self.fc1 = torch.nn.Linear(D_in, H)
		self.fc2 = torch.nn.Linear(H, H)
		self.fc3 = torch.nn.Linear(H, D_out)
		self.bn1 = torch.nn.BatchNorm1d(H)
		self.bn2 = torch.nn.BatchNorm1d(H)
		self.bn3 = torch.nn.BatchNorm1d(D_out)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
	
		h_relu = self.relu(self.bn1(self.fc1(x).clamp(min=0)))
		h_relu = self.relu(self.bn2(self.fc2(h_relu).clamp(min=0)))
		Es_pred = self.relu(self.bn3(self.fc3(h_relu)))
		return Es_pred


class EcNet(torch.nn.module):

	# Input: features
	# Output: e^{c,i}, a^i for agent i


	def __init__(self, D_in, H, D_out):
		
		super(EcNet, self).__init__()
		self.fc1 = torch.nn.Linear(D_in, H)
		self.fc2 = torch.nn.Linear(H, H)
		self.fc31 = torch.nn.Linear(H, D_out)
		self.fc32 = torch.nn.Linear(H, D_out)
		self.bn1 = torch.nn.BatchNorm1d(H)
		self.bn2 = torch.nn.BatchNorm1d(H)
		self.bn31 = torch.nn.BatchNorm1d(D_out)
		self.bn32 = torch.nn.BatchNorm1d(D_out)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
	
		h_relu = self.relu(self.bn1(self.fc1(x).clamp(min=0)))
		h_relu = self.relu(self.bn2(self.fc2(h_relu).clamp(min=0)))
		e_c = self.relu(self.bn31(self.fc31(h_relu)))
		a = self.relu(self.bn32(self.fc32(h_relu)))
		return e_c, a


def poolingFeature(agentID, list_attentions, list_Ecs, list_Ess):
	self_attention = list_attentions[agentID]
	softmax = torch.nn.Softmax(dim=1)
	#others_attentions = list_attentions.remove(self_attention)
	
	# others_Ecs = np.copy(list_Ecs)
	# del others_Ecs[agentID]

	# others_Ess = np.copy(list_Ess)
	# del others_Ess[agentID]

	diff_attention = [(np.linalg.norm(self_attention - x))^2 for x in list_attentions]


	#normalizing_const = np.sum(diff_attention)

	weights = [softmax(y) for y in diff_attention]
	weights[agentID] = 0.0

	P = [weights[i]*list_Ecs[i][j] for i in range(np.shape(list_Ecs)[0]) for j in range(len(list_Ecs[i]))]


	return np.sum(P)

'''
def softmax(y, normalizing_const):
	return np.exp(y)/normalizing_const
'''



class NetForTermination(torch.nn.Module):
	def __init__(self, num_agents, D_in_1, H1, D_out_1, D_in_2, H2, D_out_2, D_in_3, H3, D_out_3, D_in_4, H4, D_out_4):
	
		super(NetForTermination, self).__init__()


	def forward(self, x):

		list_attentions = []
		list_Ecs = []
		list_Ess = []



		# each agent comutes his prediction of other agents termination
		for i in range(num_agents):

			block1 = FeatureExtractNet(D_in_1, H1, D_out_1)
			features = block1(x)

			block2 = EsNet(D_in_2, H2, D_out_2)
			e_s = block2(features) # self encoding vector

			list_Ess.append(e_s)


			block3 = EcNet(D_in_3, H3, D_out_3)
			e_c, a = block3(x)


			list_Ecs.append(e_c)
			list_attentions.append(a)


		P = np.zeros(num_agents)
		for i in range(num_agents):
			P[i] = poolingFeature(i, list_attentions, list_Ecs, list_Ess)	


		C = [(0.0,0.0) for _ in range(num_agents)]
		decoderBlock = EsNet(D_in_4, H4, D_out_4) 
		out_pred = [[] for _ in range(num_agents)]

		for i in range(num_agents):
			C[i] = (P[i],list_Ecs[i])

		# I am not sure what the decoder net will look like. I am reusing EsNet for now.

			y = decoderBlock(C[i]) # y is a 2x1 array

			normalizing_const2 = np.sum(np.exp(y))

			# Forward pass: Compute predicted y by passing x to the model
			out_pred[i] = [softmax(y[i], normalizing_const2) for j in range(len(y))]







# Main code

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in_1, H1, D_out_1, D_in_2, H2, D_out_2, D_in_3, H3, D_out_3, D_in_4, H4, D_out_4  = 64, 1000, 100, 10, 1000, 100, 10, 1000, 100, 10, 1000, 100, 10

num_agents = 3


# Input is the joint state drawn from belief
x = torch.randn(N, D_in_1) 

# Output is a list of lists holding the termination prob vectors for each agent j 
output = [[0.1,0.9], [0.4,0.6],[0.8,0.2]]



criterion = torch.nn.MSELoss(reduction='sum') 

block1 = FeatureExtractNet(D_in_1, H1, D_out_1)
block2 = EsNet(D_in_2, H2, D_out_2)
block3 = EcNet(D_in_3, H3, D_out_3)
decoderBlock = EsNet(D_in_4, H4, D_out_4)


Net = [block1,block2,block3,decoderBlock]
parameters = set()
for net_ in Net:
parameters |= set(net_.parameters())

optimizer = torch.optim.SGD(Net.parameters(), lr=1e-4)


for t in range(500):
    
	out_pred = NetForTermination(num_agents, D_in_1, H1, D_out_1, D_in_2, H2, D_out_2, D_in_3, H3, D_out_3, D_in_4, H4, D_out_4)
    # Compute and print loss
    loss = criterion(out_pred, output)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

	













		




