from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning

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
        self.b = MultinomialDirichletBelief(env, initial_joint_observation)
        #self.b0 = Belief(env)

        '''
        3. Sample a joint state s := vec(s_1,...,s_n) according to b_0
        '''
        self.s = self.b.sampleJointState()

        # policy over options
        self.mu_policy = mu_policy

        self.o = self.chooseOption()
        self.a = self.chooseAction()

        
    def chooseOption(self):
        # Choose joint-option o based on softmax option-policy
        
        #select agents randomly to pick options
        agent_order = [agent.ID for agent in self.env.agents]
        shuffle(agent_order)
        print('agent order :', agent_order)
    
        #let agents select options from available option pool
        for agent in agent_order:
            option_mask = [not(option.available) for option in self.options]
            # print(option_mask)
    
            # pmf = [0, 0, 0.7, 0.1, 0.2]
            pmf = self.mu_policy.pmf(self.s[agent])
            pmf = np.ma.masked_array(pmf, option_mask)
            # print('pmf : ', pmf)

            # select option for agent
            # TODO : in order to sample option instead of choosing the best one, the masked pdf needs to be re-normalized
            selected_option_idx = np.argmax(pmf)
            self.env.agents[agent].option = self.options[selected_option_idx].optionID
            # print(selected_option_idx)

            #remove the selected option from available option pool by setting availability to False
            self.options[selected_option_idx].available = False

    def chooseAction(self):
        joint_action = []
        for agent in self.env.agents:
            action = self.options[agent.option].policy.sample(agent.state)
            print('agent ID:', agent.ID, 'state:', agent.state, 'option ID:', agent.option, 'action:', action)
            agent.action = action
            joint_action.append(action)

        return joint_action


 

    def evaluateOption(self, critic, action_critic, terminations, baseline=False):
        # critic.start(joint_state, joint_option)
        # action_critic.start(joint_state, joint_option, joint_action)
        
        reward, next_true_joint_state, done, _ = self.env.step(joint_action)

        broadcasts = Broadcast(self.env, next_true_joint_state, self.s, self.o, terminations)

        #broadcasts = self.env.broadcast(reward, next_true_joint_state, self.s, self.o, terminations)
        joint_observation = self.env.get_observation(broadcasts)

        self.b = MultinomialDirichletBelief(self.env, joint_observation)
        self.s = self.b.sampleJointState()

        # Critic update
        update_target = critic.update(self.s, self.o, reward, done)
        action_critic.update(self.s, self.o, self.a, reward, done)


        critic_feedback = action_critic.getQvalue(self.s, self.o, self.a)  #Q(s,o,a)

        if baseline:
            critic_feedback -= critic.value(self.s, self.o)
        return critic_feedback


    def improveOption_of_agent(self, agentID, intra_option_policy_improvement, termination_improvement, critic_feedback):
        return intra_option_policy_improvement.update(agent_state, agent_action, critic_feedback), termination_improvement.update(agentID, self.s, self.o)