from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning
from optionCritic.policies import SoftmaxOptionPolicy, SoftmaxActionPolicy

from distributed.belief import MultinomialDirichletBelief
from distributed.broadcast import Broadcast
from random import shuffle
import numpy as np


class DOC:
    def __init__(self, env, options, mu_policy):
        '''
        :param states_list: all combination of joint states. This is an input from the environment
        :param lr_thea: list of learning rates for learning policy parameters (pi), for all the agents
        :param lr_phi: list of learning rates for learning termination functions (beta), for all the agents
        :param init_observation: list of joint observation of all the agents
        '''

        self.env = env
        self.options = options
    
        '''
        2. Start with initial common belief b
        '''
        # set initial belief
        initial_joint_observation = params['env']['initial_joint_state']
        self.belief = MultinomialDirichletBelief(env, initial_joint_observation)
        #self.b0 = Belief(env)

        '''
        3. Sample a joint state s := vec(s_1,...,s_n) according to b_0
        '''
        self.joint_state = self.belief.sampleJointState()
        print('joint_state',self.joint_state)

        # policy over options
        self.mu_policy = mu_policy
        #print(self.mu_policy)

        self.joint_option = self.chooseOption() #self.chooseOption()
        self.joint_action = self.chooseAction(self.joint_state,self.joint_option)


    def chooseOption(self):
        # Choose joint-option o based on softmax option-policy
        joint_state = tuple(np.sort(self.joint_state))

        joint_option = self.mu_policy.sample(joint_state)
        print('joint_option',joint_option)

        for option in self.options:
            option.available = True

        for option in joint_option:
            self.options[option].available = False

        return joint_option

#     def chooseAction(self):
#         joint_action = []
#         for agent in self.env.agents:
#             print('agent state', agent.state, 'agent option', agent.option)
#             action = self.options[agent.option].policy.sample(agent.state)
#             print('agent ID:', agent.ID, 'state:', agent.state, 'option ID:', agent.option, 'action:', action)
#             agent.action = action
#             joint_action.append(action)

#         return joint_action

    def chooseAction(self, joint_state, joint_option):
        joint_action = []
        for agent in self.env.agents:
            print('agent state', agent.state, 'agent option', agent.option)
            agent.state = joint_state[agent.ID]
            agent.option = joint_option[agent.ID]
            agent_action = self.options[agent.option].policy.sample(agent.state)
            print('agent ID:', agent.ID, 'state:', agent.state, 'option ID:', agent.option, 'agent action:', agent_action)
            agent.action = agent_action
            joint_action.append(agent_action)

        return joint_action


 

    def evaluateOption(self, critic, action_critic, terminations, baseline=False):
        # critic.start(joint_state, joint_option)
        # action_critic.start(joint_state, joint_option, joint_action)
        
        reward, next_true_joint_state, done, _ = self.env.step(joint_action)

        broadcasts = Broadcast(self.env, next_true_joint_state, self.joint_state, self.joint_option,done).broadcastBasedOnQ(critic,reward)

        #broadcasts = self.env.broadcast(reward, next_true_joint_state, self.s, self.o, terminations)
        joint_observation = self.env.get_observation(broadcasts)

        self.belief = MultinomialDirichletBelief(self.env, joint_observation)
        self.joint_state = self.belief.sampleJointState()

        # Critic update
        update_target = critic.update(self.joint_state, self.joint_option, reward, done)
        action_critic.update(self.joint_state, self.joint_option, self.joint_action, reward, done)


        critic_feedback = action_critic.getQvalue(self.joint_state, self.joint_option, self.joint_action)  #Q(s,o,a)

        if baseline:
            critic_feedback -= critic.value(self.joint_state, self.joint_option)
        return critic_feedback


    def improveOption_of_agent(self, agentID, intra_option_policy_improvement, termination_improvement, critic_feedback):
        return intra_option_policy_improvement.update(agent_state, agent_action, critic_feedback), termination_improvement.update(agentID, self.joint_state, self.joint_option)