from modelConfig import params

class TerminationGradient:
    def __init__(self, agent_options, critic, lr=params['train']['lr_phi']):
        self.terminations = [opt.termination for opt in agent_options]
        self.critic = critic
        self.lr = lr

    def update(self, phi, option):
        magnitude, direction = self.terminations[option].grad(phi)
        self.terminations[option].weights[direction] -= \
                self.lr*magnitude*(self.critic.advantage(phi, option))


class IntraOptionGradient:
    def __init__(self, agent_options, lr=params['train']['lr_theta']):
        self.lr = lr
        self.agent_pi_policies = [option.policy for option in agent_options]


    def update(self, phi, option, action, critic):
        agent_action_pmf = self.agent_pi_policies[option].pmf(phi)
        self.agent_pi_policies[option].weights[phi, :] -= self.lr*critic*agent_action_pmf
        self.agent_pi_policies[option].weights[phi, action] += self.lr*(critic - (self.agent_pi_policies[option].weights[phi, action] - 1./params['agent']['n_actions']))


class FixedActionPolicies:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]

    def sample(self, phi):
        return self.action

    def pmf(self, phi):
        return self.probs

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
# # Check this
# class IntraOptionGradient:
# 	def __init__(self, options, lr=params['train']['lr_theta']):
# 		self.lr = lr
# 		self.pi_policy = [option.policy for option in options] #as a list of options in use


# 	def update(self, agents, joint_state, joint_option, joint_action, action_critic_value):
# 		# joint state refers to sampled joint state, s_k
# 		for idx, state, option, action in zip(range(len(joint_state)), joint_state, joint_option, joint_action):
# 			agent_state = agents[idx].state
# 			log_pi = self.pi_policy[option].pmf(agent_state)
# 			self.pi_policy[option].weights[state, :] -= self.lr*action_critic_value*log_pi
# 			self.pi_policy[option].weights[state, action] += self.lr*action_critic_value
