import numpy as np
from modelConfig import params
import random
from optionCritic.Qlearning import IntraOptionQLearning
import copy

class Broadcast:
    def __init__(self, env): 
        self.env = env


    def broadcastBasedOnQ(self, critic, reward, curr_true_joint_state, prev_sampled_joint_state, prev_joint_obs, prev_true_joint_state, prev_joint_action, joint_option,done):
        self.joint_option = joint_option
        # self.terminations = terminations
        self.done = done

        self.no_broadcast_threshold = params['env']['no_broadcast_threshold']

        broadcasts = np.zeros(self.env.n_agents)
        for agent in self.env.agents:
            estimated_curr_joint_state = np.zeros(self.env.n_agents)

            other_agents = [other_agent for other_agent in self.env.agents if other_agent.ID != agent.ID]
           
            #estimate other agents' current states
            for other_agent in other_agents:
                if prev_joint_obs[other_agent.ID] is None or prev_joint_obs[other_agent.ID][1] is None:
                    
                    # estimate the next cell of agents using last sampled state and randomly chosen 
                    other_actions_for_other = [int(a) for a in self.env.actions]
                    chosen_action_for_other = np.random.choice(other_actions_for_other,1)
        
                    while self.env.occupancy[tuple(self.env.tocellcoord[prev_sampled_joint_state[other_agent.ID]]+self.env.directions[int(chosen_action_for_other)])] == 1:
                        other_actions_for_other.remove(chosen_action_for_other[0])
                        chosen_action_for_other = np.random.choice(other_actions_for_other,1) 
        
                    estimated_curr_cell_for_other = self.env.tocellcoord[prev_sampled_joint_state[other_agent.ID]] + self.env.directions[int(chosen_action_for_other)]

                    estimated_curr_joint_state[other_agent.ID] = self.env.tocellnum[tuple(estimated_curr_cell_for_other)]
                else:
                    other_actions_for_other = [int(a) for a in self.env.actions]
                    chosen_action_for_other = prev_joint_obs[other_agent.ID][1]
                    while self.env.occupancy[tuple(self.env.tocellcoord[prev_sampled_joint_state[other_agent.ID]]+self.env.directions[int(chosen_action_for_other)])] == 1:
                        # other_actions_for_other.remove(prev_joint_obs[other_agent.ID][1])
                        other_actions_for_other.remove(chosen_action_for_other)
                        chosen_action_for_other = np.random.choice(other_actions_for_other,1) 

                    estimated_curr_joint_state[other_agent.ID] = self.env.tocellnum[tuple(self.env.tocellcoord[prev_sampled_joint_state[other_agent.ID]] + self.env.directions[int(chosen_action_for_other)])]

           
            estimated_curr_joint_state[agent.ID] = agent.state

            if prev_true_joint_state[agent.ID] == prev_sampled_joint_state[agent.ID] and prev_joint_action[agent.ID] is not None: #this means agent broadcast at last instant
                estimated_own_curr_cell = self.env.tocellcoord[prev_sampled_joint_state[agent.ID]] + self.env.directions[prev_joint_action[agent.ID]]
             
            else:   
                # estimate the next cell of agent using last sampled state and randomly chosen 
                other_actions = [int(a) for a in self.env.actions]
                chosen_action = np.random.choice(other_actions,1)
                while self.env.occupancy[tuple(self.env.tocellcoord[prev_sampled_joint_state[agent.ID]]+self.env.directions[int(chosen_action)])] == 1:
                    other_actions.remove(int(chosen_action))
                    chosen_action = np.random.choice(other_actions,1) 
                estimated_own_curr_cell = self.env.tocellcoord[prev_sampled_joint_state[agent.ID]] + self.env.directions[int(chosen_action)]

            error_due_to_no_broadcast = np.linalg.norm([i-j for (i,j) in zip(estimated_own_curr_cell,self.env.tocellcoord[curr_true_joint_state[agent.ID]])])

            # broadcast current true state if the estimated next cell is a wall
         

            # modified_current_joint_state = tuple(np.sort(modified_current_joint_state))
            selfishness_penalty = params['env']['selfishness_penalty']*error_due_to_no_broadcast*(error_due_to_no_broadcast > self.no_broadcast_threshold)
            #critic = IntraOptionQLearning(params['env']['discount'], params['doc']['lr_Q'],self.terminations, option_weights)
            
            estimated_curr_joint_state = [int(item) for item in estimated_curr_joint_state] # make th entries integer

            # print('estimated_curr_joint_state', estimated_curr_joint_state)
            critic1 = copy.deepcopy(critic)
            Q_agent_with_broadcast = critic1.update(tuple(estimated_curr_joint_state), self.joint_option, reward+self.env.broadcast_penalty, self.done)
            # q1 = critic1.update(modified_current_joint_state, self.joint_option, reward+self.env.broadcast_penalty, self.done)
            # Q_agent_with_broadcast = q1.getQvalue(modified_current_joint_state, None, self.joint_option)

            critic2 =copy.deepcopy(critic)
            Q_agent_without_broadcast = critic2.update(tuple(estimated_curr_joint_state), self.joint_option, reward+selfishness_penalty, self.done)
            # q2 = critic2.update(modified_current_joint_state, self.joint_option, reward, self.done)
            # Q_agent_without_broadcast = q2.getQvalue(modified_current_joint_state, None, self.joint_option)
            broadcasts[agent.ID] = 1*((agent.state in self.env.goals) or (Q_agent_with_broadcast >= Q_agent_without_broadcast))
        return tuple(broadcasts)


    def randomBroadcast(self, state):
        n = self.rng.uniform(0,1)
        if n < 0.1:
            return 1  # broadcast with probability 0.1
        return 0