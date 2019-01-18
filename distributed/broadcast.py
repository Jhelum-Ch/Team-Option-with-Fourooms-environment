from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning

class Broadcast:
    def __init__(self, env, true_current_joint_state, past_sampled_joint_state, joint_option, terminations):
        self.env = env
        self.true_current_joint_state = true_current_joint_state
        self.past_sampled_joint_state = past_sampled_joint_state
        self.joint_option = joint_option
        self.terminations = terminations

    # def broadcastBasedOnQ(self, Q0, Q1):
    #     """An agent broadcasts if the agent is at any goal or the intra-option value for
    #     no broadcast (Q0) is less than that with broadcast (Q1)"""

    #     return (self.agent.state in self.goals) or (Q0 < Q1)


    def broadcastBasedOnQ(self, reward):
        broadcasts = np.zeros(self.n_agents)
        for agent in self.agents:
            modified_current_joint_state = np.copy(self.past_sampled_joint_state)
            modified_current_joint_state[agent] = self.true_current_joint_state[agent]


            critic = IntraOptionQLearning(params['env']['discount'], params['doc']['lr_Q'],terminations)
            critic1 = copy.deepcopy(critic)
            q1 = critic1.update(modified_current_joint_state, joint_option, reward+env.broadcast_penalty, done)
            Q_agent_with_broadcast = q1.getQvalue(modified_current_joint_state, None, joint_option)


            critic2 =copy.deepcopy(critic)
            q2 = critic2.update(modified_current_joint_state, joint_option, reward, done)
            Q_agent_without_broadcast = q2.getQvalue(modified_current_joint_state, None, joint_option)

            broadcasts[agent] = 1*((self.env.agent.state in self.env.goals) or (Q_agent_with_broadcast > Q_agent_without_broadcast))
        return tuple(broadcasts)
    


    def randomBroadcast(self, state):
        n = self.rng.uniform(0,1)
        if n < 0.1:
            return 1  # broadcast with probability 0.1
        return 0