# import torch

# """
# A fully-connected ReLU network with two hidden layers, trained to predict y from x
# by minimizing squared Euclidean distance.
# This implementation uses the nn package from PyTorch to build the network.
# PyTorch autograd makes it easy to define computational graphs and take gradients,
# but raw autograd can be a bit too low-level for defining complex neural networks;
# this is where the nn package can help. The nn package defines a set of Modules,
# which you can think of as a neural network layer that has produces output from
# input and may have some trainable weights or other state.
# """

# #device = torch.device('cpu')
# # device = torch.device('cuda') # Uncomment this to run on GPU

# # N is batch size; D_in is input dimension;
# # H1 and H2 are the dimensions of two hidden layers; D_out is output dimension.
# N, D_in, H1, H2, D_out = 64, 1000, 100, 100, 10

# # Create random Tensors to hold inputs and outputs
# # x = torch.randn(N, D_in, device=device)
# # y = torch.randn(N, D_out, device=device)

# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # Use the nn package to define our model as a sequence of layers. nn.Sequential
# # is a Module which contains other Modules, and applies them in sequence to
# # produce its output. Each Linear Module computes output from input using a
# # linear function, and holds internal Tensors for its weight and bias.
# # After constructing the model we use the .to() method to move it to the
# # desired device.
# # model = torch.nn.Sequential(
# #           torch.nn.Linear(D_in, H1),
# #           torch.nn.ReLU(),
# #           torch.nn.Linear(H1, H2),
# #           torch.nn.ReLU(),
# #           torch.nn.Linear(H2, D_out),
# #         ).to(device)

# model = torch.nn.Sequential(
#           torch.nn.Linear(D_in, H1),
#           torch.nn.ReLU(),
#           torch.nn.Linear(H1, H2),
#           torch.nn.ReLU(),
#           torch.nn.Linear(H2, D_out),
#         )

# # The nn package also contains definitions of popular loss functions; in this
# # case we will use Mean Squared Error (MSE) as our loss function. Setting
# # reduction='sum' means that we are computing the *sum* of squared errors rather
# # than the mean; this is for consistency with the examples above where we
# # manually compute the loss, but in practice it is more common to use mean
# # squared error as a loss by setting reduction='elementwise_mean'.
# # loss_fn = torch.nn.MSELoss(reduction='sum')


# learning_rate = 1e-4
# for t in range(500):
# 	# Forward pass: compute predicted y by passing x to the model. Module objects
# 	# override the __call__ operator so you can call them like functions. When
# 	# doing so you pass a Tensor of input data to the Module and it produces
# 	# a Tensor of output data.
# 	y_pred = model(x)
# 	loss_fn = (y_pred - y).pow(2).sum()
# 	print(t, loss.item())

# 	# Compute and print loss. We pass Tensors containing the predicted and true
# 	# values of y, and the loss function returns a Tensor containing the loss.
# 	loss = loss_fn(y_pred, y)
# 	print(t, loss.item())

# 	# Zero the gradients before running the backward pass.
# 	model.zero_grad()

# 	# Backward pass: compute gradient of the loss with respect to all the learnable
# 	# parameters of the model. Internally, the parameters of each Module are stored
# 	# in Tensors with requires_grad=True, so this call will compute gradients for
# 	# all learnable parameters in the model.
# 	loss.backward()

# 	# Update the weights using gradient descent. Each parameter is a Tensor, so
# 	# we can access its data and gradients like we did before.
# 	with torch.no_grad():
# 		for param in model.parameters():
		  # param.data -= learning_rate * param.grad



# # -*- coding: utf-8 -*-
# import torch

# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Create random Tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # Use the nn package to define our model as a sequence of layers. nn.Sequential
# # is a Module which contains other Modules, and applies them in sequence to
# # produce its output. Each Linear Module computes output from input using a
# # linear function, and holds internal Tensors for its weight and bias.
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )

# # The nn package also contains definitions of popular loss functions; in this
# # case we will use Mean Squared Error (MSE) as our loss function.
# loss_fn = torch.nn.MSELoss(reduction='sum')

# learning_rate = 1e-4
# for t in range(500):
#     # Forward pass: compute predicted y by passing x to the model. Module objects
#     # override the __call__ operator so you can call them like functions. When
#     # doing so you pass a Tensor of input data to the Module and it produces
#     # a Tensor of output data.
#     y_pred = model(x)

#     # Compute and print loss. We pass Tensors containing the predicted and true
#     # values of y, and the loss function returns a Tensor containing the
#     # loss.
#     loss = loss_fn(y_pred, y)
#     print(t, loss.item())

#     # Zero the gradients before running the backward pass.
#     model.zero_grad()

#     # Backward pass: compute gradient of the loss with respect to all the learnable
#     # parameters of the model. Internally, the parameters of each Module are stored
#     # in Tensors with requires_grad=True, so this call will compute gradients for
#     # all learnable parameters in the model.
#     loss.backward()

#     # Update the weights using gradient descent. Each parameter is a Tensor, so
#     # we can access its gradients like we did before.
#     with torch.no_grad():
#         for param in model.parameters():
#             param -= learning_rate * param.grad

# -*- coding: utf-8 -*-
# import torch


# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)

#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred


# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Create random Tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # Construct our model by instantiating the class defined above
# model = TwoLayerNet(D_in, H, D_out)

# # Construct our loss function and an Optimizer. The call to model.parameters()
# # in the SGD constructor will contain the learnable parameters of the two
# # nn.Linear modules which are members of the model.
# criterion = torch.nn.MSELoss(reduction='mean') #reduction='sum'
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
# for t in range(500):
#     # Forward pass: Compute predicted y by passing x to the model
#     y_pred = model(x)

#     # Compute and print loss
#     loss = criterion(y_pred, y)
#     print(t, loss.item())

#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
