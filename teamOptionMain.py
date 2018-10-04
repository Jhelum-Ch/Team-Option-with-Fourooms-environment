import gym
import argparse
import numpy as np
from fourroomsMA import FourroomsMA

from scipy.special import expit
from scipy.misc import logsumexp
import dill

class Tabular:
    def __init__(self, n_states):
        self.n_states = n_states

    def __call__(self, state): #state is a tuple having numAgents elements
        return np.array([state,])

    def __len__(self):
        return self.n_states


class SoftmaxPolicy:
	def __init__(self, numAgents, rng, n_features, n_actions, temp=1.):
	    self.numAgents = numAgents
	    self.rng = rng
	    self.weights = np.zeros((n_features, n_actions)) # we assume that n_features and n_actions for all agents are same
	    self.temp = temp

	def value(self, Phi, joint_action=[None for _ in range(self.numAgents)]): #Phi is a n_features x numAgents dimensional matrix, joint_action is a list 
	    value = np.zeros(self.numAgents)
	    for i in range(self.numAgents):
		    if joint_action[i] is None:
		        value[i] = np.sum(self.weights[Phi[:,i],:], axis=0)
	    	value[i] = np.sum(self.weights[Phi[:,i], joint_action[i]], axis=0)
    	return value


	def pmf(self, Phi):
		pmf = np.zeros(self.numAgents)
		for i in range(self.numAgents):
	    	v = self.value(Phi[:,i])/self.temp
	    	pmf[i] = np.exp(v - logsumexp(v))
	    return pmf


	def sample(self, Phi):
		sample_jointAction = np.zeros(self.numAgents)
		for i in range(self.numAgents):
			sample_jointAction[i] = int(self.rng.choice(self.weights.shape[1], p=self.pmf(Phi[:,i])))
	    return sample_jointAction


class EgreedyPolicy:
    def __init__(self, numAgents, rng, n_features, n_actions, epsilon):
        self.numAgents = numAgents
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((n_features, n_actions))

    def value(self, Phi, joint_action=[None for _ in range(self.numAgents)]):
    	value = np.zeros(self.numAgents)
    	for i in range(self.numAgents)
	        if joint_action[i] is None:
	            value[i] = np.sum(self.weights[Phi[:,i], :], axis=0)
        	value[i] = np.sum(self.weights[Phi[:,i], joint_action[i]], axis=0)
    	return value

    def sample(self, Phi):
    	sample_jointAction = np.zeros(self.numAgents)
    	for i in range(self.numAgents):
			if self.rng.uniform() < self.epsilon:
            	sample_jointAction[i] = int(self.rng.randint(self.weights.shape[1]))
        	sample_jointAction[i] = int(np.argmax(self.value(phi))) 
    	return sample_jointAction


class SigmoidTermination:
    def __init__(self, numAgents, rng, n_features):
        self.numAgents = numAgents
        self.rng = rng
        self.weights = np.zeros((n_features,))

    def pmf(self, Phi):
    	pmf = np.zeros(self.numAgents)
    	for i in range(self.numAgents):
    		pmf[i] = expit(np.sum(self.weights[Phi[:,i]]))
        return pmf

    def sample(self, Phi):
    	sample_terminate = np.zeros(self.numAgents)
    	for i in range(self.numAgents):
    		sample_terminate[i] = int(self.rng.uniform() < self.pmf(Phi[:,i]))
        return sample_terminate

    def grad(self, Phi): # Check this formula
        terminate = self.pmf(Phi)
        return [p*(1. - p) for p in terminate], Phi


class IntraOptionQLearning:
    def __init__(self, numAgents, discount, lr, broadcast, terminations, weights):
        self.numAgents = numAgents
        self.discount = discount
        self.lr = lr
        self.broadcast = broadcast
        self.terminations = terminations
        self.weights = weights

    def start(self, Phi, joint_option): #Phi is a matrix and joint_option is a list
        self.last_Phi = Phi
        self.last_jointOption = joint_option 
        self.last_value = self.value(Phi, joint_option)

    def value(self, Phi, joint_option): #Some of joint_option values could be None
	    out = np.zeros(self.numAgents)
	    for i in range(self.numAgents):
	        if joint_option[i] is None:
	            out[i,:] = np.sum(self.weights[Phi[:,i], :], axis=0)
	        out[i,:] = np.sum(self.weights[Phi[:,i], joint_option[i]], axis=0)

        return np.sum(out, axis=0) #add value of each agent to get total value. Returns an array for all actions


    # Double check the following function
    def advantage(self, Phi, joint_option): #Some of options values could be None
        values = self.value(Phi, joint_option)
        advantages = values - np.max(values)
        for i in range(self.numAgents):
	        if joint_option[i] is None:
	            return advantages
            advantages[joint_option[i]]
        return 

    def update(self, Phi, joint_option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(broadcast, Phi, joint_option)
            termination = self.terminations[self.last_jointOption].pmf(Phi)
            not_all_termination_prob = np.prod(np.ones(self.numAgents) - termination) 
            update_target += self.discount*(not_all_termination_prob*current_values[self.last_jointOption] + (1.-not_all_termination_prob)*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_Phi, self.last_jointOption] += self.lr*tderror

        if not done:
            self.last_value = current_values[joint_option]
        self.last_jointOption = joint_option
        self.last_Phi = Phi

        return update_target


class IntraOptionActionQLearning:
    def __init__(self, numAgents, discount, lr, terminations, weights, qbigomega):        
        self.numAgents = numAgents
        self.discount = discount
        self.lr = lr
        self.terminations = terminations #terminations is a list 
        self.weights = weights
        self.qbigomega = qbigomega

    def value(self, Phi, joint_option, joint_action): #Phi is a matrix, joint_option and joint_action are lists
    	out = np.zeros(self.numAgents)
    	for i in range(self.numAgents):
    		out[i] = np.sum(self.weights[Phi[:,i], joint_option[i], joint_action[i]], axis=0)

        return np.sum(out)

    def start(self, Phi, joint_option, joint_action): #Phi is a matrix, joint_option and joint_action are lists
        self.last_Phi = Phi
        self.last_jointOption = joint_option
        self.last_jointAction = joint_action

    def update(self, Phi, joint_option, joint_action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(Phi)
            termination = self.terminations[self.last_options].pmf(Phi)
            not_all_termination_prob = np.prod(np.ones(self.numAgents) - termination) 
            update_target += self.discount*(not_all_termination_prob*current_values[self.last_jointOption] + (1. - not_all_termination_prob)*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_Phi, self.last_jointOption, self.last_jointAction)
        self.weights[self.last_Phi, self.last_jointOption, self.last_jointAction] += self.lr*tderror

        self.last_Phi = Phi
        self.last_jointOption = joint_option
        self.last_jointAction = joint_action



#finish the following four classes

class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, Phi, joint_option):
        magnitudes, directions = self.terminations[joint_option].grad(Phi)
        self.terminations[options].weights[directions] -= \
                self.lr*magnitudes*(self.critic.advantage(Phi, joint_option))


class IntraOptionGradient:
    def __init__(self, pi_policies, lr):
        self.lr = lr
        self.pi_policies = pi_policies

    def update(self, Phi, joint_option, joint_action, critic):
        joint_actionPmf = self.pi_policies[joint_option].pmf(Phi)
        self.pi_policies[joint_option].weights[Phi, :] -= self.lr*critic*joint_actionPmf
        self.pi_policies[joint_option].weights[Phi, joint_action] += self.lr*critic




class OneStepTermination:
    def sample(self, Phi):
        return [1 for _ in range(np.shape(Phi)[1])]

    def pmf(self, Phi):
        return [1. for _ in range(np.shape(Phi)[1])]


class FixedActionPolicies:
    def __init__(self, joint_action, n_actions):
        self.joint_action = joint_action
        self.probs = np.eye(n_actions)[joint_action]

    def sample(self, phi):
        return self.action

    def pmf(self, phi):
        return self.probs



		



#___________________


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nagents', help='Number of agents', type=int, default=1)
    parser.add_argument('--ngoals', help='Number of goals', type=int, default=1)
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=1e-2)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-2)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=250)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true', default=False)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')

    args = parser.parse_args()

    rng = np.random.RandomState(1234)
    env = gym.make('FourroomsMA-v0')

    fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
    fname = 'optioncritic-fourroomsMA-' + fname + '.npy'

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]


	history = np.zeros((args.nruns, args.nepisodes, args.nagents+1))
	    for run in range(args.nruns):
	        features = Tabular(env.observation_space.n)
	        nfeatures, nactions = len(features), env.action_space.n #len(features) = 1
	        #nfeatures, nactions = len(features), env.numAgents**env.action_space.n #len(features) = 1

	        # The intra-option policies are linear-softmax functions
	        pi_policies = [SoftmaxPolicy(rng, nfeatures, nactions, args.temperature) for _ in range(args.noptions)]
	        if args.primitive:
	            pi_policies.extend([FixedActionPolicies(act, nactions) for act in range(nactions)])


	        # The termination function are linear-sigmoid functions
	        option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(args.noptions)]
	        if args.primitive:
	            option_terminations.extend([OneStepTermination() for _ in range(nactions)])

	        # Softmax policy over options
	        #mu_policy = EgreedyPolicy(rng, args.nagents, nfeatures, args.noptions, args.epsilon)
	        mu_policy = SoftmaxPolicy(rng, args.nagents, nfeatures, args.noptions, args.temperature)

	        # Different choices are possible for the critic. Here we learn an
	        # option-value function and use the estimator for the values upon arrival
	        critic = IntraOptionQLearning(args.nagents, args.discount, args.lr_critic, option_terminations, policy.weights)

	        # Learn Qomega separately
	        action_weights = np.zeros((nfeatures, args.noptions, nactions))
	        action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights, critic)

	        # Improvement of the termination functions based on gradients
	        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term)

	        # Intra-option gradient improvement with critic estimator
	        intraoption_improvement = IntraOptionGradient(pi_policies, args.lr_intra)

	        for episode in range(args.nepisodes):
	            if episode == 1000:
	                env.goals = rng.choice(possible_next_goals, arg.ngoals, replace=False)
	                print('************* New goals : ', env.goal)

	            Phi = features(env.reset())
	            joint_option = [mu_policy.sample(Phi[:,i]) for i in range(env.numAgents)]
	            joint_action = [pi_policies[joint_option[i]].sample(Phi[:,i]) for i in range(env.numAgents)]
	            critic.start(Phi, joint_option)
	            action_critic.start(Phi, joint_option, joint_action)



	            cumreward = 0.
	            duration = 1
	            option_switches = [0 for _ in env.numAgents]
	            avgduration = [0. for _ in env.numAgents]
	            broadcasts = [agent.actions.broadcast for agent in env.agents]
				phi0 = np.zeros(n_features)
				phi1 = mp.copy(phi0)


	            for step in range(args.nsteps):
	                observations, reward, done, _ = env.step(actions,broadcasts) #observations is a list of tuples of (state, action) of each agent
	                observation_states = [x[0] for x in observations] #list of states of each agent in the list observations
	               
	                
	                sample_states = env.sample_from_belief(self,observation_states,broadcasts)
	            	Phi = features(sample_states) #Phi is a matrix
	                broadcasts0 = np.copy(broadcasts)
	            	broadcasts1 = np.copy(broadcasts)

	                # Termination might occur upon entering the new state
	                option_terminationList = option_terminations[joint_option].sample(Phi)
	               	joint_action = pi_policies[joint_option].sample(Phi)
	                for i, agent in enumerate(env.agents):
		                if option_terminationList[i]:
		                    joint_option[i] = mu_policy.sample(Phi[:,i])  
		                    option_switches[i] += 1
		                    avgduration[i] += (1./option_switches[i])*(duration - avgduration[i])
		                    duration = 1
	               		agent.actions.action = joint_action[i]
	                     

	                    
			            broadcasts0[i] = 0.
			            broadcasts1[i] = 1.
			            sample_states_from_belief0 = env.sample_from_belief(self,observation_states,broadcasts0)
			            sample_states_from_belief1 = env.sample_from_belief(self,observation_states,broadcasts1)
			            phi0 = features(sample_states_from_belief0)[:,i]
			            phi1 = features(sample_states_from_belief1)[:,i]
			            Q0 = critic.update(phi0, joint_option, reward, done) 
			            Q1 = critic.update(phi1, joint_option, reward, done) 

	                    agent.actions.broadcast = env.broadcast(agent, Q0, Q1)


	                # Critic update
	                update_target = critic.update(Phi, joint_option, reward, done)
	                action_critic.update(Phi, joint_option, joint_action, reward, done)

	                if isinstance(pi_policies[joint_option], SoftmaxPolicy):
	                    # Intra-option policy update
	                    critic_feedback = action_critic.value(Phi, joint_option, joint_action)
	                    if args.baseline:
	                        critic_feedback -= critic.value(Phi, joint_option)
	                    intraoption_improvement.update(Phi, joint_option, joint_action, critic_feedback)

	                    # Termination update
	                    termination_improvement.update(Phi, joint_option)


	                cumreward += reward
	                duration += 1
	                if done:
	                    break

	            history[run, episode, 0] = step
	            history[run, episode, [1:end] = avgduration
	            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))
	        np.save(fname, history)
	        dill.dump({'intra_optionPolicies':pi_policies, 'mu_policy':mu_policy, 'term':option_terminations}, open('oc-options.pl', 'wb'))
	        print(fname)













