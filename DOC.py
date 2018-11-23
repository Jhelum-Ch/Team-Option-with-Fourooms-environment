import argparse
import random

import dill
import gym
import numpy as np
from OptionCritic import *

from agent import Agent
from belief import Belief
from fourroomsEnv import FourroomsMA


class Tabular:
    def __init__(self, n_states, n_agents):
        self.n_states = n_states**n_agents

    def __call__(self, joint_state): #joint state is a tuple having n_agents elements
        total = 0.0
        for i in range(len(joint_state)):
            total += joint_state[i]*(10.0**(len(joint_state)-i-1))
        return np.array([total,])

    def __len__(self):
        return self.n_states


class Broadcast:
    def __init__(self, goals, agent, optionID):
        self.goals = goals
        self.agent = agent
        self.optionID = optionID

    def broadcastBasedOnQ(self, Q0, Q1):    
        """An agent broadcasts if the agent is at any goal or the intra-option value for 
        no broadcast (Q0) is less than that with broadcast (Q1)"""

        return (agent.state in self.goals) or (Q0 < Q1)


    def randomBroadcast(self, state):
        n = self.rng.uniform(0,1)
        if n < 0.1:
            return 1  # broadcast with probability 0.1
        return 0


#___________________


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', help='Number of agents', type=int, default=1)
    parser.add_argument('--n_goals', help='Number of goals', type=int, default=1)
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=1e-2)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-2)
    parser.add_argument('--n_episodes', help="Number of episodes per run", type=int, default=250)
    parser.add_argument('--n_runs', help="Number of runs", type=int, default=100)
    parser.add_argument('--n_steps', help="Maximum number of steps per episode", type=int, default=1000)
    parser.add_argument('--n_options', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true', default=False)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')

    args = parser.parse_args()

    rng = np.random.RandomState(1234)
    env = gym.make('FourroomsMA-v0')
    agents = [Agent() for _ in range(args.n_agents)]

    fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
    fname = 'optioncritic-fourroomsMA-' + fname + '.npy'

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]


    print("TESTING")
    x = SoftmaxPolicy(rng, 5, 5, args.temperature)
    print(x)


    history = np.zeros((args.n_runs, args.n_episodes, args.n_agents+1))
    for run in range(args.n_runs):
        # Initialize for every run
        features = Tabular(env.observation_space.n)
        n_features, n_actions = len(features), env.action_space.n #len(features) = 1

        belief = Belief(args.n_agents, states_list)


        # The intra-option policies are linear-softmax functions
        policies = [SoftmaxPolicy(rng, n_features, n_actions, args.temperature) for _ in available_optionIDs]
        if args.primitive:
            policies.extend([FixedActionPolicies(act, n_actions) for act in range(n_actions)])


        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, n_features) for _ in available_terminationIDs]
        if args.primitive:
            option_terminations.extend([OneStepTermination() for _ in range(n_actions)])

        # Softmax policy over options
        #mu_policy = EgreedyPolicy(rng, args.nagents, nfeatures, args.noptions, args.epsilon)
        mu_policy = SoftmaxPolicy(rng, n_features, n_options_available, args.temperature)


        available_optionIDs = [i for i in range(args.n_options)] #option IDs for available options
        n_options_available = len(available_optionIDs)
        available_terminationIDs = [i for i in range(args.n_options)] #termination IDs for available terminations
        n_terminations_available = len(available_terminationIDs)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        option_critic = IntraOptionQLearning(args.n_agents, args.discount, args.lr_critic, option_terminations, mu_policy.weights)

        # Learn Qomega separately
        action_weights = np.zeros((n_features, n_options_available, n_actions))
        action_critic = IntraOptionActionQLearning(args.n_agents, args.discount, args.lr_critic, option_terminations, action_weights, option_critic)

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, option_critic, args.lr_term)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(policies, args.lr_intra)

        for episode in range(args.nepisodes):
            if episode == 1000:
                env.goals = rng.choice(possible_next_goals, arg.ngoals, replace=False)
                print('************* New goals : ', env.goal)

            phi = features(env.reset())


            available_optionIDs = [i for i in range(args.n_options)] #option IDs for available options
            n_options_available = len(available_optionIDs)
            available_terminationIDs = [i for i in range(args.n_options)] #termination IDs for available terminations
            n_terminations_available = len(available_terminationIDs)


            # Check how to sample from Mu-policy distribution without replacement
            joint_optionIDs = random.sample(available_optionIDs, k = args.n_agents) # All agents sample IDs from pool of available options without replacement

            #joint_option = [mu_policy.sample(Phi[:,i]) for i in range(joint_optionIDs)]
            #joint_action = [pi_policies[joint_option[i]].sample(Phi[:,i]) for i in range(args.n_agents)]
            joint_action = [policies[joint_optionIDs[i]].sample(phi) for i in range(len(joint_optionIDs))]
            option_critic.start(phi, joint_option)
            action_critic.start(phi, joint_option, joint_action)



            cumreward = 0.
            duration = 1
            option_switches = [0 for _ in range(args.n_agents)]
            avgduration = [0. for _ in range(args.n_agents)]
            broadcasts = [agent.Option().broadcast for agent in agents]

            # The following two are required to check for broadcast for each agent in every step
            phi0 = np.zeros(n_features)
            phi1 = mp.copy(phi0)


            observation_samples = np.zeros((belief.N, args.n_agents,))
            for i in range(belief.N):
                for j in range(args.n_agents):
                    observation_samples[i,j] = env.currstate[j]

            for step in range(args.n_steps):

                # 1. Start with a pool of available options and terminations
                available_optionIDs = [x for x in available_optionIDs if x not in joint_optionIDs]
                available_terminationIDs = [x for x in available_terminationIDs if x not in joint_optionIDs]


                # 2. Sample a joint_state (tuple) from the initial belief
                joint_state = belief.sample_single_state()

                # The following two are required to check for broadcast for each agent in every step
                joint_state_with_broadcast = np.copy(joint_state)
                joint_state_without_broadcast = np.copy(joint_state)


                # 3. Compute actual actions
                actual_joint_actions = [a.option.policy.sample_action(joint_state) for a in agents]

                # 4. Get reward from environment based on on actual actions
                reward, done, _ = env.step(actual_joint_action)

               # The following two are required to check for broadcast for each agent in every step
                reward_with_broadcast = np.copy(reward)
                reward_without_broadcast = np.copy(reward)

                # sample_observations = belief.sample()
                # update_belief = belief.updateBeliefParameters(sample_observations)


               # 5. For all agents, determine if they broadcast
                observation_samples_agent_with_braodcast = np.zeros_like(observation_samples)
                observation_samples_agent_without_braodcast = np.zeros_like(observation_samples)

                for i in range(args.n_agents):
                    a = agents[i]
                    if broadcasts[i] is None:
                        broadcasts[i] = Broadcast().chooseBroadcast(next_state, a.option.ID)


                    reward_with_broadcast  += env.broadcast_penalty
                    observation_samples_agent_with_braodcast[:,i] =  joint_state_with_broadcast[i]*np.ones(np.shape(observation_samples)[0])
                    newBelief_with_broadcast = belief.updateBeliefParameters(observation_samples_agent_with_braodcast)
                    next_sampleJointState_with_broadcast = newBelief_with_broadcast.sample_single_state()
                    phi1 = features(next_sampleJointState_with_broadcast)[:,i]
                    Q1 = critic.update(phi1, joint_option, reward_with_broadcast, done)


                    newBelief_without_broadcast = belief.updateBeliefParameters(observation_samples_agent_without_braodcast)
                    next_sampleJointState_without_broadcast = newBelief_without_broadcast.sample_single_state()
                    phi0 = features(next_sampleJointState_without_broadcast)[:,i]
                    Q0 = critic.update(phi0, joint_option, reward_without_broadcast, done)


                    broadcasts[i] = Broadcast(goals, a, joint_optionIDs[i]).broadcastBasedOnQ(a, Q0, Q1)


                # 6. Get observation based on broadcasts
                y = env.get_observation(broadcasts)


                # 7. Decrease reward based on number of broadcasting agents
                reward += np.sum(broadcasts)*env.broadcast_penalty


                # 8. Termination might occur upon entering the new state. Then, choose a new option from the pool pf available options
                option_terminationList = [option_terminations[joint_optionIDs[i]].sample(phi) for i in range(args.n_agents)]
                joint_action = policies[joint_optionIDs].sample(phi)

                n_terminated_agents = np.sum(1.0*option_terminationList)

                for i, agent in enumerate(agents):
                    terminated_agentIDs = []
                    if option_terminationList[i]:
                        broadcasts[i] = 1.0 #force broadcast on termination
                        option_switches[i] += 1
                        avgduration[i] += (1./option_switches[i])*(duration - avgduration[i])
                        duration = 1
                        terminated_agentIDs.append(i)
                        available_optionIDs.append(joint_optionIDs[i]) #terminated option becomes available
                        n_options_available = len(available_optionIDs)
                        available_terminationIDs.append(joint_optionIDs[i]) #terminated option becomes available

                # Terminated agents simultaneously choose without replacement the option IDs from the pool of
                #available options
                new_optionIDs_for_terminated_agents = random.sample(available_optionIDs, k = n_terminated_agents) # Check how to sample without replacement from mu_policy distribution

                # New joint option
                for i, j in enumerate(terminated_agentIDs):
                    joint_optionIDs[j] = new_optionIDs_for_terminated_agents[i]
                    joint_action[j] = policies[joint_optionIDs[j]].sample(phi)




                # Critic update
                update_target = critic.update(phi, joint_option, reward, done)
                action_critic.update(phi, joint_option, joint_action, reward, done)

                if isinstance(policies[joint_option], SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = action_critic.value(phi, joint_option, joint_action)
                    if args.baseline:
                        critic_feedback -= critic.value(phi, joint_option)
                    intraoption_improvement.update(phi, joint_option, joint_action, critic_feedback)

                    # Termination update
                    termination_improvement.update(phi, joint_option)


                cumreward += reward
                duration += 1
                if done:
                    break

            history[run, episode, 0] = step
            #history[run, episode, [1:end] = avgduration
            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))
        np.save(fname, history)
        dill.dump({'intra_optionPolicies':policies, 'term':option_terminations}, open('oc-options.pl', 'wb'))
        print(fname)













