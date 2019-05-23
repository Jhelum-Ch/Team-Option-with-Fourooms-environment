

from modelConfig import params

# class TerminationGradient:
# 	def __init__(self, options, critic, lr=params['train']['lr_phi']):
# 		self.termination = [option.termination for option in options]
# 		self.critic = critic
# 		self.lr = lr

# 	def update(self, next_joint_state, estimated_next_joint_state, joint_option):
# 		# joint_state refers to sampled next state (estimated_next_state), s_k^'
# 		advantage = self.critic.getAdvantage(estimated_next_joint_state, joint_option)	 + params['train'][
# 			'deliberation_cost']
# 		for state, option in zip(next_joint_state, joint_option):
# 			# phi = joint_state[agentID]
# 			magnitudes, directions = self.termination[option].grad(state)
# 			self.termination[option].weights[directions] -= \
# 					self.lr*magnitudes*advantage
		

# Check this
class IntraOptionGradient:
	def __init__(self, options, lr=params['train']['lr_theta']):
		self.lr = lr
		self.pi_policy = []
		for agent_idx in range(params['env']['n_agents']):
			self.pi_policy.append([option.policy for option in options[agent_idx]]) #as a list of options in use


	def update(self, agents, joint_state, joint_option, joint_action, action_critic_value):
		# joint state refers to sampled joint state, s_k
		# for idx, state, option, action in zip(range(len(joint_state)), joint_state, joint_option, joint_action):
		for idx, (state, option, action) in enumerate(zip(joint_state, joint_option, joint_action)):
			# agent_state = agents[idx].state
			#pi = SoftmaxActionPolicy(self.env.cell_list,params['agent']['n_actions'])
			log_pi = self.pi_policy[idx][option].pmf(state)
			self.pi_policy[idx][option].weights[state, :] -= self.lr*action_critic_value*log_pi
			self.pi_policy[idx][option].weights[state, action] += self.lr*(action_critic_value - (self.pi_policy[idx][option].weights[state, action] - 1./params['agent']['n_actions']))

