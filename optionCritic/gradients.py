# from policies import SoftmaxPolicy, EgreedyPolicy, FixedActionPolicies
# from termination import SigmoidTermination, OneStepTermination


# class TerminationGradient:
#     def __init__(self, terminations, critic, lr):
#         self.terminations = terminations
#         self.critic = critic
#         self.lr = lr

#     def update(self, phi, joint_optionID,joint_option): #use joint-state in place of phi
#         magnitudes, directions = self.terminations[joint_optionID].grad(phi)
#         self.terminations[joint_optionID].weights[directions] -= \
#                 self.lr*magnitudes*(self.critic.advantage(phi, joint_option))



# # Check this
# class IntraOptionGradient:
#     def __init__(self, pi_policies, lr):
#         self.lr = lr
#         self.pi_policies = pi_policies

#     def update(self, phi, joint_optionID, joint_option, joint_action, critic): #use joint-state in place of phi
#         joint_actionPmf = self.pi_policies[joint_option].pmf(phi)
#         self.pi_policies[joint_optionID].weights[phi, :] -= self.lr*critic*joint_actionPmf
#         self.pi_policies[joint_optionID].weights[phi, joint_action] += self.lr*critic

from modelConfig import params

class TerminationGradient:
	def __init__(self, options, critic, lr=params['train']['lr_phi']):
		self.termination = [option.termination for option in options]
		self.critic = critic
		self.lr = lr

	def update(self, next_joint_state, estimated_next_joint_state, joint_option):
		# joint_state refers to sampled next state (estimated_next_state), s_k^'
		advantage = self.critic.getAdvantage(estimated_next_joint_state, joint_option)	 + params['train'][
			'deliberation_cost']
		for state, option in zip(next_joint_state, joint_option):
			# phi = joint_state[agentID]
			magnitudes, directions = self.termination[option].grad(state)
			self.termination[option].weights[directions] -= \
					self.lr*magnitudes*advantage
# Check this
class IntraOptionGradient:
	def __init__(self, options, lr=params['train']['lr_theta']):
		self.lr = lr
		self.pi_policy = [option.policy for option in options] #as a list of options in use


	def update(self, agents, joint_state, joint_option, joint_action, action_critic_value):
		# joint state refers to sampled joint state, s_k
		for idx, state, option, action in zip(range(len(joint_state)), joint_state, joint_option, joint_action):
			agent_state = agents[idx].state
			log_pi = self.pi_policy[option].pmf(agent_state)
			self.pi_policy[option].weights[state, :] -= self.lr*action_critic_value*log_pi
			self.pi_policy[option].weights[state, action] += self.lr*action_critic_value
