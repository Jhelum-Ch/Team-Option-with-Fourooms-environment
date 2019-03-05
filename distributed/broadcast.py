import numpy as np
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning
import copy

class Broadcast:
    def __init__(self, env): #terminations
        self.env = env
        # self.true_current_joint_state = true_current_joint_state
        # self.past_sampled_joint_state = past_sampled_joint_state
        # self.joint_option = joint_option
        # self.terminations = terminations
        # self.done = done

    # def broadcastBasedOnQ(self, Q0, Q1):
    #     """An agent broadcasts if the agent is at any goal or the intra-option value for
    #     no broadcast (Q0) is less than that with broadcast (Q1)"""

    #     return (self.agent.state in self.goals) or (Q0 < Q1)


    # def broadcastBasedOnQ(self, critic, reward, next_true_joint_state, sampled_curr_joint_state, joint_option,done):
    #     #self.next_true_joint_state = next_true_joint_state
    #     #self.sampled_curr_joint_state = sampled_curr_joint_state
    #     self.joint_option = joint_option
    #     # self.terminations = terminations
    #     self.done = done
    #     # should take agentQ instead of critic
    #     broadcasts = np.zeros(self.env.n_agents)
    #     for agent in self.env.agents:
    #         modified_current_joint_state = np.copy(sampled_curr_joint_state)
    #         modified_current_joint_state[agent.ID] = next_true_joint_state[agent.ID]
    #         # modified_current_joint_state = tuple(np.sort(modified_current_joint_state))

    #         #critic = IntraOptionQLearning(params['env']['discount'], params['doc']['lr_Q'],self.terminations, option_weights)
    #         critic1 = copy.deepcopy(critic)
    #         Q_agent_with_broadcast = critic1.update(modified_current_joint_state, self.joint_option, reward+self.env.broadcast_penalty, self.done)
    #         # q1 = critic1.update(modified_current_joint_state, self.joint_option, reward+self.env.broadcast_penalty, self.done)
    #         # Q_agent_with_broadcast = q1.getQvalue(modified_current_joint_state, None, self.joint_option)

    #         critic2 =copy.deepcopy(critic)
    #         Q_agent_without_broadcast = critic2.update(modified_current_joint_state, self.joint_option, reward, self.done)
    #         # q2 = critic2.update(modified_current_joint_state, self.joint_option, reward, self.done)
    #         # Q_agent_without_broadcast = q2.getQvalue(modified_current_joint_state, None, self.joint_option)

    #         broadcasts[agent.ID] = 1*((agent.state in self.env.goals) or (Q_agent_with_broadcast > Q_agent_without_broadcast))
    #     return tuple(broadcasts)


    def broadcastBasedOnQ(self, critic, reward, curr_true_joint_state, sampled_curr_joint_state, joint_option,done):
        #self.next_true_joint_state = next_true_joint_state
        #self.sampled_curr_joint_state = sampled_curr_joint_state
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
            #print(tuple(estimated_next_cell), (tuple(estimated_next_cell) in self.env.states_list))

            error_due_to_no_broadcast = np.linalg.norm([i-j for (i,j) in zip(estimated_next_cell,self.env.tocellcoord[curr_true_joint_state[agent.ID]])])

            # broadcast current true state if the estimated next cell is a wall
            #if self.env.occupancy[tuple(estimated_next_cell)] != 0:
            if tuple(estimated_next_cell) not in self.env.states_list:
                #error_due_to_no_broadcast += -0.05
                broadcasts[agent.ID] = 1

            else:
                # modified_current_joint_state = tuple(np.sort(modified_current_joint_state))
                selfishness_penalty = params['env']['selfishness_penalty']*error_due_to_no_broadcast*(error_due_to_no_broadcast > self.no_broadcast_threshold)
                #critic = IntraOptionQLearning(params['env']['discount'], params['doc']['lr_Q'],self.terminations, option_weights)
                critic1 = copy.deepcopy(critic)
                Q_agent_with_broadcast = critic1.update(modified_current_joint_state, self.joint_option, reward+self.env.broadcast_penalty, self.done)
                # q1 = critic1.update(modified_current_joint_state, self.joint_option, reward+self.env.broadcast_penalty, self.done)
                # Q_agent_with_broadcast = q1.getQvalue(modified_current_joint_state, None, self.joint_option)

                critic2 =copy.deepcopy(critic)
                Q_agent_without_broadcast = critic2.update(modified_current_joint_state, self.joint_option, reward+selfishness_penalty, self.done)
                # q2 = critic2.update(modified_current_joint_state, self.joint_option, reward, self.done)
                # Q_agent_without_broadcast = q2.getQvalue(modified_current_joint_state, None, self.joint_option)
                #print('agent', agent.ID, 'estimated_sample_state', self.env.tocellnum[tuple(estimated_next_cell)], 'true_state', curr_true_joint_state[agent.ID], 'error', error_due_to_no_broadcast, 'Qwb_agent', Q_agent_with_broadcast, 'Qwob_agent', Q_agent_without_broadcast)
                broadcasts[agent.ID] = 1*((agent.state in self.env.goals) or (Q_agent_with_broadcast >= Q_agent_without_broadcast))
        return tuple(broadcasts)


    def randomBroadcast(self, state):
        n = self.rng.uniform(0,1)
        if n < 0.1:
            return 1  # broadcast with probability 0.1
        return 0