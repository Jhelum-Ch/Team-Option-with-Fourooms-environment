import numpy as np

class IntraOptionQLearning:
    def __init__(self, n_agents, discount, lr, terminations, weights):
        self.n_agents = n_agents
        self.discount = discount
        self.lr = lr
        self.terminations = terminations
        self.weights = weights

    def start(self, phi, joint_option): #phi is a scalar of all agents and joint_option is a list
        self.last_phi = phi
        self.last_jointOption = joint_option
        self.last_value = self.value(phi, joint_option)

    def value(self, phi, joint_option): #Some of joint_option values could be None
        for i in range(len(joint_option)):
            if joint_option[i] is None:
                out[i,:] = np.sum(self.weights[phi, :], axis=0)
            out[i,:] = np.sum(self.weights[phi, joint_option[i]], axis=0)

        return np.sum(out, axis=0) #add value of each agent to get total value. Returns an array for all actions

    def one_or_more_terminate_prob(self, n_agents, terminations):
        superset = [list(combinations(range(n_agents)))]
        superset.remove(set())

        sumtotal = 0.0
        for item in superset:
            product = 1.0
            for i in item:
                product *= terminations[item[i]]

            sumtotal += product

        return sumtotal

    # TODO: Double check the following function
    def advantage(self, phi, joint_option): #Some of options values could be None
        values = self.value(Phi, joint_option)
        advantages = values - np.max(values)
        for i in range(self.numAgents):
            if joint_option[i] is None:
                return advantages
            advantages[joint_option[i]]
        return

    def update(self, phi, joint_option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi, joint_option)
            termination = self.terminations[self.last_jointOption].pmf(phi)

            #modify this according to current writeup
            one_or_more_termination_prob = self.one_or_more_terminate_prob(self.n_agents, self.terminations)
            update_target += self.discount*((1.-one_or_more_termination_prob)*current_values[self.last_jointOption] + one_or_more_termination_prob*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_Phi, self.last_jointOption] += self.lr*tderror

        if not done:
            self.last_value = current_values[joint_option]
            self.last_jointOption = joint_option
            self.last_Phi = phi

        return update_target


class IntraOptionActionQLearning:
    def __init__(self, n_agents, discount, lr, terminations, weights, qbigomega):
        self.n_agents = n_agents
        self.discount = discount
        self.lr = lr
        self.terminations = terminations #terminations is a list
        self.weights = weights
        self.qbigomega = qbigomega

    def value(self, phi, joint_option, joint_action): #Phi is a matrix, joint_option and joint_action are lists
        out[i] = np.sum(self.weights[phi, joint_option[i], joint_action[i]], axis=0)

        return np.sum(out)

    def one_or_more_terminate_prob(self, n_agents, terminations):
        superset = [list(combinations(range(n_agents)))]
        superset.remove(set())

        sumtotal = 0.0
        for item in superset:
            product = 1.0
            for i in item:
                product *= terminations[item[i]]

            sumtotal += product

        return sumtotal

    def start(self, phi, joint_option, joint_action): #Phi is a matrix, joint_option and joint_action are lists
        self.last_Phi = phi
        self.last_jointOption = joint_option
        self.last_jointAction = joint_action

    def update(self, phi, joint_option, joint_action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_options].pmf(Phi)
            one_or_more_termination_prob = self.one_or_more_terminate_prob(self.n_agents, self.terminations)
            update_target += self.discount*((1.-one_or_more_termination_prob)*current_values[self.last_jointOption] + one_or_more_termination_prob*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_Phi, self.last_jointOption, self.last_jointAction)
        self.weights[self.last_Phi, self.last_jointOption, self.last_jointAction] += self.lr*tderror

        self.last_Phi = phi
        self.last_jointOption = joint_option
        self.last_jointAction = joint_action