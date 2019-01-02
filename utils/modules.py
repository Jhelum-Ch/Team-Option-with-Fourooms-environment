import numpy as np
from modelConfig import params
from optionCritic.option import Option
from optionCritic.policies import SoftmaxPolicy

def createOptions(env):
    options = []
    for i in range(params['agent']['n_options']):
        options.append(Option(i, SoftmaxPolicy(env)))
        
    return options

class StateEncoder:
    '''
    input :     joint state of agents
                represented as tuple
                example : (1, 2, 3) where agent1 is at state 1, agent2 is at state 2 and agent3 is at state3
    output :    concatenated unique joint state
                example : (1, 2, 3) --> 123
    note : Any unique representation of states would work
    '''
    def __init__(self, n_states, n_agents):
        self.n_states = n_states**n_agents

    def __call__(self, joint_state): #joint state is a tuple having n_agents elements
        total = 0
        for i in range(len(joint_state)):
            total += joint_state[i]*(10**(len(joint_state)-i-1))
        return np.array([total,])

    def __len__(self):
        return self.n_states