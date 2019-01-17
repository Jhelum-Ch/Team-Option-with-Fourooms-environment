# from policies import SoftmaxPolicy, EgreedyPolicy, FixedActionPolicies
# from termination import SigmoidTermination, OneStepTermination


class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, joint_state, joint_option):
        for option in joint_option:
            magnitudes, directions = self.terminations[option].grad(joint_state)
            self.terminations[option].weights[directions] -= \
                    self.lr*magnitudes*(self.critic.getAdvantage(joint_state, joint_option))



# Check this
class IntraOptionGradient:
    def __init__(self, pi_policies, lr):
        self.lr = lr
        self.pi_policies = pi_policies

    def update(self, joint_state, joint_option, joint_action, critic):
        joint_actionPmf = self.pi_policies[joint_option].pmf(joint_state)
        self.pi_policies[joint_option].weights[phi, :] -= self.lr*critic*joint_actionPmf
        self.pi_policies[joint_option].weights[phi, joint_action] += self.lr*critic
