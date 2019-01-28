import numpy as np
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning
import copy

class Broadcast:
    def __init__(self, env, true_current_joint_state, past_sampled_joint_state, joint_option,done): #terminations
        self.env = env
        self.true_current_joint_state = true_current_joint_state
        self.past_sampled_joint_state = past_sampled_joint_state
        self.joint_option = joint_option
        #self.terminations = terminations
        self.done = done

    # def broadcastBasedOnQ(self, Q0, Q1):
    #     """An agent broadcasts if the agent is at any goal or the intra-option value for
    #     no broadcast (Q0) is less than that with broadcast (Q1)"""

    #     return (self.agent.state in self.goals) or (Q0 < Q1)


    def broadcastBasedOnQ(self, critic, reward):
        broadcasts = np.zeros(self.env.n_agents)
        for agent in self.env.agents:
            modified_current_joint_state = np.copy(self.past_sampled_joint_state)
            modified_current_joint_state[agent.ID] = self.true_current_joint_state[agent.ID]


            #critic = IntraOptionQLearning(params['env']['discount'], params['doc']['lr_Q'],self.terminations, option_weights)
            critic1 = copy.deepcopy(critic)
            Q_agent_with_broadcast = critic1.update(modified_current_joint_state, self.joint_option, reward+self.env.broadcast_penalty, self.done)
            # q1 = critic1.update(modified_current_joint_state, self.joint_option, reward+self.env.broadcast_penalty, self.done)
            # Q_agent_with_broadcast = q1.getQvalue(modified_current_joint_state, None, self.joint_option)


            critic2 =copy.deepcopy(critic)
            Q_agent_without_broadcast = critic2.update(modified_current_joint_state, self.joint_option, reward, self.done)
            # q2 = critic2.update(modified_current_joint_state, self.joint_option, reward, self.done)
            # Q_agent_without_broadcast = q2.getQvalue(modified_current_joint_state, None, self.joint_option)

            broadcasts[agent.ID] = 1*((agent.state in self.env.goals) or (Q_agent_with_broadcast > Q_agent_without_broadcast))
        return tuple(broadcasts)
    


    def randomBroadcast(self, state):
        n = self.rng.uniform(0,1)
        if n < 0.1:
            return 1  # broadcast with probability 0.1
        return 0