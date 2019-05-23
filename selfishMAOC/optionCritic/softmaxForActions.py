import numpy as np
from scipy.misc import logsumexp
from modelConfig import params
from modelConfig import params


class SoftmaxPolicy_for_action:
    def __init__(self, weights, temp=params['policy']['temperature']):
        '''
        :param temp: lower temperature means uniform distribution, higher means delta
        '''
        self.rng = params['train']['seed']

        self.weights = weights
        # weights is a dictionary keeping track of Q(s,o,a) values. This is the weight dictionary of IntraOptionAction Q
        # learning
        self.temp = temp

    def getQvalue(self, agent_state, agent_option, action=None):
        # 1. select all rows where agent_state and agent_option appear
        agent_state_option_keys = [(s,o) for s in self.weights.keys() for o in self.weights[s].keys() if agent_state in s and agent_option in o]
        # print(agent_state_option_keys)
        # print(len(agent_state_option_keys))

        # 2. Make the for each tion:
        # 		for each row:
        # 			calculate the sum of Q values where option appears in joint_options for that row

        Q = np.zeros((params['agent']['n_actions']))

        for option in range(params['agent']['n_options']):
            for (joint_state,joint_option) in agent_state_option_keys:
                # print(joint_state,joint_option)
                action_keys = [a for a in self.weights[joint_state][joint_option].keys() if action in a]
                # print(action_keys)
                for a in action_keys:
                    Q[action] += self.weights[joint_state][joint_option][a]	#TODO : need to average instead of sum
        # print(Q)

        # TODO : mask Q for unavailable options

        return Q


    def pmf(self, agent_state, agent_option):
        v = self.getQvalue(agent_state, agent_option) / self.temp
        pmf = np.exp(v - logsumexp(v))
        return pmf

    def sample(self, joint_state, joint_option, agent_action):
        joint_action = []
        for agent_state, agent_option in joint_state, joint_option:
            sample_agent_action = int(self.rng.choice(range(params['agent']['n_actions']), p=self.pmf(agent_state, agent_option)))
            joint_action.append(sample_agent_action)
            #TODO : make the sampled option unavailable
        return tuple(joint_action)