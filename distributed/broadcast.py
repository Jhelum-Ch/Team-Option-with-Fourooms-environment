import numpy as np
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning
import copy

class Broadcast:
    def __init__(self, env): #terminations
        self.env = env

    def broadcastBasedOnQ(self, critic, reward, curr_true_joint_state, sampled_curr_joint_state, joint_option,done):
        self.joint_option = joint_option
        # self.terminations = terminations
        self.done = done

        self.no_broadcast_threshold = params['env']['no_broadcast_threshold']
        # should take agentQ instead of critic
        broadcasts = np.zeros(self.env.n_agents)
        for agent in self.env.agents:
            modified_current_joint_state = np.copy(sampled_curr_joint_state) # this is sampled joint state from last instant
            modified_current_joint_state[agent.ID] = curr_true_joint_state[agent.ID]

            estimated_next_cell = self.env.tocellcoord[sampled_curr_joint_state[agent.ID]] + self.env.directions[agent.action]

            error_due_to_no_broadcast = np.linalg.norm([i-j for (i,j) in zip(estimated_next_cell,self.env.tocellcoord[curr_true_joint_state[agent.ID]])])

            # broadcast current true state if the estimated next cell is a wall
            #if self.env.occupancy[tuple(estimated_next_cell)] != 0:
            if tuple(estimated_next_cell) not in self.env.states_list:
                #error_due_to_no_broadcast += -0.05
                broadcasts[agent.ID] = 1

            else:
                selfishness_penalty = params['env']['selfishness_penalty']*error_due_to_no_broadcast*(error_due_to_no_broadcast > self.no_broadcast_threshold)
         
                critic1 = copy.deepcopy(critic)
                Q_agent_with_broadcast = critic1.update(modified_current_joint_state, self.joint_option, reward+self.env.broadcast_penalty, self.done)
              
                critic2 =copy.deepcopy(critic)
                Q_agent_without_broadcast = critic2.update(modified_current_joint_state, self.joint_option, reward+selfishness_penalty, self.done)
                broadcasts[agent.ID] = 1*((agent.state in self.env.goals) or (Q_agent_with_broadcast >= Q_agent_without_broadcast))
        return tuple(broadcasts)


    def randomBroadcast(self, state):
        n = self.rng.uniform(0,1)
        if n < 0.1:
            return 1  # broadcast with probability 0.1
        return 0