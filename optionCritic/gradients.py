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



class TerminationGradient:
    def __init__(self, termination, critic, lr):
        self.termination = termination
        self.critic = critic
        self.lr = lr

    def update(self, agentID, joint_state, joint_option): #use joint-state in place of phi
        phi = joint_state[agentID]
        magnitudes, directions = self.termination.grad(phi)
        self.termination.weights[directions] -= \
                self.lr*magnitudes*(self.critic.getAdvantage(joint_state, joint_option))
# Check this
class IntraOptionGradient:
    def __init__(self, pi_policy, lr):
        self.lr = lr
        self.pi_policy = pi_policy

    def update(self, phi, agent_action, critic): # phi is agent_state
        agent_actionPmf = self.pi_policy.pmf(phi)
        self.pi_policy.weights[phi, :] -= self.lr*critic*agent_actionPmf
        self.pi_policy.weights[phi, agent_action] += self.lr*critic