"""
Skeleton Main code to meant to run the workflow. Incomplete in that it lacks elements from the Option-Critic architecture
"""

import argparse
import random

import dill
import gym
import numpy as np

from optionCritic.gradients import IntraOptionGradient, TerminationGradient
from optionCritic.policies import SoftmaxPolicy, FixedActionPolicies
from optionCritic.termination import SigmoidTermination, OneStepTermination
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning

from distributed.broadcast import Broadcast
from distributed.belief import Belief

from option import Option
from agent import Agent
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


#___________________


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', help='Number of agents', type=int, default=3)
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
    parser.add_argument('--primitive', help="Augment with primitive", default=True, action='store_true')

    args = parser.parse_args()

    rng = np.random.RandomState(1234)

    # TODO:  since the code makes a lot of calls to internal attributes of the environment,
    # TODO:  initializing with gym.make doesn't work
    #env = gym.make('FourroomsMA-v0')
    env = FourroomsMA(n_agents=args.n_agents)

    fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
    fname = 'optioncritic-fourroomsMA-' + fname + '.npy'

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]

    history = np.zeros((args.n_runs, args.n_episodes, args.n_agents+1))
    for run in range(args.n_runs):
        # Initialize for every run

        features = Tabular(env.observation_space.n, args.n_agents)
        n_features, n_actions = len(features), env.action_space.n   #len(features) = 1

        belief = Belief(args.n_agents, env.states_list)

        # # This commented block is unverified
        # JC-comment: I have uncommented the intra-option policies and termination function

        # available_optionIDs = [i for i in range(args.n_options)]  # option IDs for available options
        # n_options_available = len(available_optionIDs)
        # available_terminationIDs = [i for i in range(args.n_options)]  # termination IDs for available terminations
        # n_terminations_available = len(available_terminationIDs)
        #
        #
        # The intra-option policies (pi_policies) are linear-softmax functions
        policies = [SoftmaxPolicy(rng, n_features, n_actions, args.temperature) for _ in available_optionIDs]
        if args.primitive:
            policies.extend([FixedActionPolicies(act, n_actions) for act in range(n_actions)])
        
        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, n_features) for _ in available_terminationIDs]
        if args.primitive:
            option_terminations.extend([OneStepTermination() for _ in range(n_actions)])


        options = [Option() for _ in range(args.n_options)]
        # tracks available options (IDs only)
        free_optionIDs = [opt_id for opt_id in range(args.n_options)]

        opt_id = 0
        for o in options:
            o.policy = SoftmaxPolicy(rng, n_features, n_actions, args.temperature)
            o.termination = SigmoidTermination(rng, n_features)
            o.ID = opt_id
            opt_id += 1

        if args.primitive:
            primitive_options = [Option() for _ in range(n_actions)]
            i = 0
            actions = list(range(n_actions))
            for o in primitive_options:
                o.policy = FixedActionPolicies(actions[i], n_actions)
                o.termination = OneStepTermination()
                o.ID = opt_id
                opt_id += 1
                i += 1

            options.extend(primitive_options)
            n_options = len(free_optionIDs)
            free_optionIDs.extend([a+n_options for a in actions])


        # Softmax policy over options
        #mu_policy = EgreedyPolicy(rng, args.nagents, nfeatures, args.noptions, args.epsilon)
        mu_policy = SoftmaxPolicy(rng, n_features, len(options), args.temperature)

        option_terminations = [o.termination for o in options]

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        option_critic = IntraOptionQLearning(args.n_agents, args.discount, args.lr_critic, option_terminations, mu_policy.weights)

        # Learn Qomega separately
        action_weights = np.zeros((n_features, len(options), n_actions))
        action_critic = IntraOptionActionQLearning(args.n_agents, args.discount, args.lr_critic, option_terminations, action_weights, option_critic)

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, option_critic, args.lr_term)

        policies = [o.policy for o in options]

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(policies, args.lr_intra)


        for episode in range(args.n_episodes):
            if episode == 1000:
                env.goals = rng.choice(possible_next_goals, args.ngoals, replace=False)
                print('************* New goals : ', env.goals)

            phi = features(env.reset())

            # ----------- Tested up to here -------------
            # TODO: Continue debugging here.


            # Check how to sample from Mu-policy distribution without replacement
            start_optionIDs = rng.choice(free_optionIDs, env.n_agents, replace=False)

            # Remove sampled options from the free ones
            for opt in start_optionIDs:
                free_optionIDs.remove(opt)

            for i in range(env.n_agents):
                env.agents[i].option = options[start_optionIDs[i]]

            option_belief = start_optionIDs.copy()

            # joint_option = [a.option for a in env.agents]       # Actual options, not IDs.

            joint_action = [o.policy.sample(phi)]

            # Initialize critic
            # TODO: Check if initialization is correct. Especially, check if need to pass actual options or option IDs
            option_critic.start(phi, start_optionIDs)
            action_critic.start(phi, start_optionIDs, joint_action)

            # Most up-to-date information on which options the agents are running
            option_belief = [None]*args.n_agents

            # Initialize tracking variables
            cumreward = 0.
            duration = 1
            option_switches = [0 for _ in range(args.n_agents)]
            avgduration = [0. for _ in range(args.n_agents)]
            broadcasts = [0 for _ in range(args.n_agents)]


            # The following two are required to check for broadcast for each agent in every step
            phi0 = np.zeros(n_features)
            phi1 = np.copy(phi0)

            for step in range(args.n_steps):

                # 1. Sample s_t ~ b_t
                state = belief.sample_single_state()


                # 2. Sample a_t = [ a_t^j ~ pi^j(s_t) ]
                actions = [a.option.policy.sample_action(state) for a in env.agents]

                # 2.1 Simulate actions a_t based on most up do date information on running options
                simu_actions = [options[ID].policy.sample_action(state) for ID in option_belief]


                # 3. Compute s_t+1 = s_t + a_t assuming that the environment layout is known and that the transition is deterministic
                # Handles wall collisions but not agent collisions
                next_state = [None for _ in range(env.n_agents)]

                for i in range(env.n_agents):
                    next_cell = tuple(env.tocellcoord[state[i]] + env.directions[simu_actions[i]])
                    if env.occupancy[next_cell] == 0:
                        next_state[i] = env.tocellnum[next_cell]
                    else:
                        next_state[i] = state[i]

                # 4. Enact actions on reward
                step_reward, done, _ = env.step(actions)

                # 5. Check option termination and assign new options. Also force broadcast on termination
                broadcasts = [None] * env.n_agents
                for i in range(env.n_agents):
                    a = env.agents[i]
                    if a.option.termination.sample(next_state) == 1:
                        optID = a.option.ID
                        free_optionIDs.append(optID)
                        a.option = None
                        broadcasts[i] = 1   # force broadcast

                        # Broadcast the option that *just* terminated, i.e. *not* the new one
                        option_belief[i] = optID

                # Sort free (available) option IDs.
                free_optionIDs.sort()

                for a in env.agents:
                    if a.option is None:

                        # TODO: Sample new options without replacement from mu_policy. 
                        new_opt = mu_policy.sample(phi)
                        free_optionIDs.remove(new_opt)
                        a.option = options[new_opt]



                # TODO: Check following two lines along with (***) in loop below. I am not certain what they do.
                # JC_comment: broadcast affects the reward, because of the penalty and the observation y. So, each agent 
                # compares the Q-values with and without broadcast to decide whether or not to broadcast the current state
                # and option ID. #6. should output the broadcast action taking into consideration aforementioned Q-values 
                observation_samples_agent_with_braodcast = np.zeros_like(observation_samples)
                observation_samples_agent_without_braodcast = np.zeros_like(observation_samples)

                # 6. For all agents, determine if they broadcast
                for i in range(env.n_agents):
                    a = env.agents[i]
                    if broadcasts[i] is None:
                        B = Broadcast().chooseBroadcast(next_state, a.option.ID)
                        broadcasts[i] = B

                        if B == 1:
                            # Update option belief based on non-forced broadcast (i.e. not a termination)
                            option_belief[i] = optID

                    # TODO: (***) Check following labelled block of text:

                    # --------From here----------

                    reward_with_broadcast += env.broadcast_penalty
                    observation_samples_agent_with_braodcast[:, i] = joint_state_with_broadcast[i] * np.ones(
                        np.shape(observation_samples)[0])
                    newBelief_with_broadcast = belief.updateBeliefParameters(observation_samples_agent_with_braodcast)
                    next_sampleJointState_with_broadcast = newBelief_with_broadcast.sample_single_state()
                    phi1 = features(next_sampleJointState_with_broadcast)[:, i]
                    Q1 = critic.update(phi1, joint_option, reward_with_broadcast, done)

                    newBelief_without_broadcast = belief.updateBeliefParameters(
                        observation_samples_agent_without_braodcast)
                    next_sampleJointState_without_broadcast = newBelief_without_broadcast.sample_single_state()
                    phi0 = features(next_sampleJointState_without_broadcast)[:, i]
                    Q0 = critic.update(phi0, joint_option, reward_without_broadcast, done)

                    # JC_comment: B = Broadcast().broadcastBasedOnQ(Q0, Q1)

                    # ----------To here----------


                # 7. Decrease reward based on number of broadcasting agents
                step_reward += np.sum(broadcasts)*env.broadcast_penalty

                # 8. Get observation based on broadcasts
                y = env.get_observation(broadcasts)

                # # 8.1 Update option information
                # # Note: If this block is UNcommented, then it means
                # # agents are broadcasting NEW option upon termination, not terminating one.
                # for i in range(env.n_agents):
                #     if broadcasts[i] == 1:
                #         option_belief[i] = env.agents[i].option.ID


                # # Initialize samples to starting state
                # if step == 0:
                #     samples = np.zeros((belief.N, env.n_agents,))
                #     for i in range(belief.N):
                #         for j in range(env.n_agents):
                #             samples[i,j] = env.currstate[j]
                # else:
                #     samples = belief.sample()


                # TODO: Right now, samples are drawn every step. Check (*) below for alternative.
                # Note: There is no "initial" batch of samples, i.e. starting joint state is unknown to the belief.
                samples = belief.sample()


                # 9. Compute data Samples based on y, by filling the gaps from the belief.
                if np.sum(broadcasts) > 0:

                    # TODO: (*) Verify if the two lines below are correct. I believe samples should be drawn every step.
                    # Samples are updated iff there was at least one broadcast
                    # samples = belief.sample()

                    for i in range(env.n_agents):
                        if y[i] is not None:
                            # Correct sampled states with broadcasted info
                            for j in range(belief.N):
                                samples[j, i] = y[i]

                            # Correct next state info (used below for termination decision)
                            next_state[i] = y[i]

                # 10. Update belief
                belief.updateBeliefParameters(samples)

                # 11. Critic update

                # TODO: Check critic update below
                # ----------From here---------

                # New (ground truth) joint option:
                joint_optionIDs = [a.option.ID for a in env.agents]
                for i, j in enumerate(terminated_agentIDs):
                    # joint_optionIDs[j] = new_optionIDs_for_terminated_agents[i]
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

                # ------------To here---------

                cumreward += step_reward
                duration += 1
                if done:
                    break

            # TODO: saving and printing below this line unverified.
            history[run, episode, 0] = step
            # history[run, episode, [1:end] = avgduration
            print(
            'Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward,
                                                                                          avgduration, option_switches))
        np.save(fname, history)
        dill.dump({'intra_optionPolicies': policies, 'term': option_terminations}, open('oc-options.pl', 'wb'))
        print(fname)
