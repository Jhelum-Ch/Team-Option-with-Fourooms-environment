import gym
import argparse
import numpy as np
import random

from fourroomsEnv import FourroomsMA
from agent import Agent
from option import Option
from belief import Belief

from scipy.special import expit
from scipy.misc import logsumexp
import dill



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


class SoftmaxPolicy:
	def __init__(self, rng, n_features, n_actions, temp=1.):
	    self.rng = rng
        self.n_actions = n_actions
	    self.weights = np.zeros((n_features, n_actions)) # we assume that n_features and n_actions for all agents are same
	    self.temp = temp

	def value(self, phi, action = None): 
	    if action is None:
	        value = np.sum(self.weights[phi,:], axis=0)
    	value = np.sum(self.weights[phi, action], axis=0)
    	return value


	def pmf(self, phi):
    	v = self.value(phi)/self.temp
    	pmf = np.exp(v - logsumexp(v))
	    return pmf


	def sample(self, phi):
		sample_action = int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))
	    return sample_action


class EgreedyPolicy:
    def __init__(self, rng, n_features, n_actions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((n_features, n_actions))

    def value(self, phi, action=None):
        if action is None:
            value = np.sum(self.weights[phi, :], axis=0)
    	value = np.sum(self.weights[phi, action], axis=0)
    	return value

    def sample(self, phi):
		if self.rng.uniform() < self.epsilon:
        	sample_action = int(self.rng.randint(self.weights.shape[1]))
    	sample_action = int(np.argmax(self.value(phi))) 
    	return sample_action


class SigmoidTermination:
    def __init__(self, rng, n_features):
        self.rng = rng
        self.weights = np.zeros((n_features,))

    def pmf(self, phi):
		pmf = expit(np.sum(self.weights[phi]))
        return pmf

    def sample(self, phi):
		sample_terminate = int(self.rng.uniform() < self.pmf(phi))
        return sample_terminate

    def grad(self, phi): # Check this formula
        terminate = self.pmf(phi)
        return [p*(1. - p) for p in terminate], phi


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


class IntraOptionQLearning:
    def __init__(self, n_agents, discount, lr, terminations, weights):
        self.n_agents = n_agents
        self.discount = discount
        self.lr = lr
        self.terminations = terminations
        self.weights = weights

    def start(self, phi, joint_option): #phi is a scalar of all agents and joint_option is a list
        self.last_phi = phi
        self.last_jointOption = joint_option 
        self.last_value = self.value(phi, joint_option)

    def value(self, phi, joint_option): #Some of joint_option values could be None
	    for i in range(len(joint_option)):
	        if joint_option[i] is None:
	            out[i,:] = np.sum(self.weights[phi, :], axis=0)
	        out[i,:] = np.sum(self.weights[phi, joint_option[i]], axis=0)

        return np.sum(out, axis=0) #add value of each agent to get total value. Returns an array for all actions

    def one_or_more_terminate_prob(self, n_agents, terminations)        
        superset = [list(combinations(range(n_agents)))]
        superset.remove(set())

        sumtotal = 0.0
        for item in superset:
            product = 1.0
            for i in item:
                product *= terminations[item[i]]

            sumtotal += product

        return sumtotal



        

    # Double check the following function
    def advantage(self, phi, joint_option): #Some of options values could be None
        values = self.value(Phi, joint_option)
        advantages = values - np.max(values)
        for i in range(self.numAgents):
	        if joint_option[i] is None:
	            return advantages
            advantages[joint_option[i]]
        return 

    def update(self, phi, joint_option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi, joint_option)
            termination = self.terminations[self.last_jointOption].pmf(phi)

            #modify this according to current writeup
            one_or_more_termination_prob = self.one_or_more_terminate_prob(self.n_agents, self.terminations) 
            update_target += self.discount*((1.-one_or_more_termination_prob)*current_values[self.last_jointOption] + one_or_more_termination_prob*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_Phi, self.last_jointOption] += self.lr*tderror

        if not done:
            self.last_value = current_values[joint_option]
            self.last_jointOption = joint_option
            self.last_Phi = phi

        return update_target


class IntraOptionActionQLearning:
    def __init__(self, n_agents, discount, lr, terminations, weights, qbigomega):        
        self.n_agents = n_agents
        self.discount = discount
        self.lr = lr
        self.terminations = terminations #terminations is a list 
        self.weights = weights
        self.qbigomega = qbigomega

    def value(self, phi, joint_option, joint_action): #Phi is a matrix, joint_option and joint_action are lists
		out[i] = np.sum(self.weights[phi, joint_option[i], joint_action[i]], axis=0)

        return np.sum(out)

    def one_or_more_terminate_prob(self, n_agents, terminations)        
        superset = [list(combinations(range(n_agents)))]
        superset.remove(set())

        sumtotal = 0.0
        for item in superset:
            product = 1.0
            for i in item:
                product *= terminations[item[i]]

            sumtotal += product

        return sumtotal

    def start(self, phi, joint_option, joint_action): #Phi is a matrix, joint_option and joint_action are lists
        self.last_Phi = phi
        self.last_jointOption = joint_option
        self.last_jointAction = joint_action

    def update(self, phi, joint_option, joint_action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_options].pmf(Phi)
            one_or_more_termination_prob = self.one_or_more_terminate_prob(self.n_agents, self.terminations) 
            update_target += self.discount*((1.-one_or_more_termination_prob)*current_values[self.last_jointOption] + one_or_more_termination_prob*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_Phi, self.last_jointOption, self.last_jointAction)
        self.weights[self.last_Phi, self.last_jointOption, self.last_jointAction] += self.lr*tderror

        self.last_Phi = phi
        self.last_jointOption = joint_option
        self.last_jointAction = joint_action



class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, phi, joint_option):
        magnitudes, directions = self.terminations[joint_option].grad(phi)
        self.terminations[joint_option].weights[directions] -= \
                self.lr*magnitudes*(self.critic.advantage(phi, joint_option))



# Check this
class IntraOptionGradient:
    def __init__(self, pi_policies, lr):
        self.lr = lr
        self.pi_policies = pi_policies

    def update(self, phi, joint_option, joint_action, critic):
        joint_actionPmf = self.pi_policies[joint_option].pmf(Phi)
        self.pi_policies[joint_option].weights[phi, :] -= self.lr*critic*joint_actionPmf
        self.pi_policies[joint_option].weights[phi, joint_action] += self.lr*critic




class OneStepTermination:
    def sample(self, phi):
        return [1 for _ in range(np.shape(phi)[1])]

    def pmf(self, phi):
        return [1. for _ in range(np.shape(phi)[1])]


class FixedActionPolicies:
    def __init__(self, joint_action, n_actions):
        self.joint_action = joint_action
        self.probs = np.eye(n_actions)[joint_action]

    def sample(self, phi):
        return self.joint_action

    def pmf(self, phi):
        return self.probs
                
                    



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
	            history[run, episode, [1:end] = avgduration
	            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))
	        np.save(fname, history)
	        dill.dump({'intra_optionPolicies':policies, 'term':option_terminations}, open('oc-options.pl', 'wb'))
	        print(fname)













