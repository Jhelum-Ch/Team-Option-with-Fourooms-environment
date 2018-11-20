import numpy as np
import fourroomsEnv
import option
import agent
import belief


'''This code does a Monte Carlo return with n agents with common belief estimated as mixGauss'''

class MCR_Gauss(env):
    
    def __init__(self, n_agents,  maxIter):
        self.n_agents = n_agents
        # self.n_mean_prior = n_mean_prior
        # self.n_cov_prior = n_cov_prior
        self.maxIter = maxIter
        super(MCR_Gauss, self).__init__()
    
    
    def mc_episode(self, features, policy, n_episodes, n_steps): 
        for ep in range(n_episodes):
            episode = []
            Q_inter = defaultdict(list)
            
            phi = features(env.reset())

            joint_action = [policy.sample(phi) for _ in range(self.n_agents)] 
            
            common_observation = env.get_observation(Option.broadcasts)
            n_samples = np.shape(common_observation)[1]

            joint_state = np.mean(Belief(self.n_agents, env.states_list).sample(), axis = 0) # But this might not yiels integer values


            
            for step in range(n_steps):
                common_observation_samples = env.get_observation(Option.broadcasts)

                
                
                common_belief_mean_posterior, common_belief_cov_posterior = Belief(self.n_agents).updateBeliefParameters(self, common_observation_samples)


                phi = features(next_joint_state)
                joint_action = [pi_policy().sample(phi) for _ in range(self.n_agents)]

                #reward = self.reward(next_joint_state, joint_action)

                reward, done = env.step(joint_action)
                next_joint_state = np.random.multivariate_normal(common_belief_mean_posterior, common_belief_cov_posterior, n_samples)

                # samples_joint_states = np.random.multivariate_normal(mean_prior, cov_prior, 100)
                # joint_state = np.mean(samples_joint_states, axis = 0) #sample_average for estimated joint-state
                next_joint_state = np.mean(Belief(self.n_agents, env.states_list).sample(), axis = 0) # But this might not yiels integer values

                episode.append([joint_state, joint_action, reward])

                if done:
                    break

                # mean_prior = common_belief_mean_posterior
                # cov_prior = common_belief_cov_posterior

                reward_from_episode = np.zeros([env.observation_space.n,env.action_space.n]) #target reward from episode
                for t1 in reversed(range(len(episode))):
                    joint_state, joint_action, reward = episode[t1]

                    reward_from_episode[joint_state,joint_action] = args.discount*reward_from_episode[joint_state,joint_action] + reward
                    Q_inter[joint_state,joint_action].append(reward_from_episode[joint_state,joint_action]) #every visit MC
                saEpisode = set([(x[0], x[1]) for x in episode]) 

                for (s,a) in saEpisode:
                    Q[s,a] = np.mean(Q_inter[(s,a)])
        return Q

    

    
class Tabular:
    def __init__(self, n_states):
        self.n_states = n_states

    def __call__(self, state): #state is a tuple having n_agents elements
        return np.array([state,])

    def __len__(self):
        return self.n_states
    

    
class SoftmaxPolicy:
    def __init__(self, rng, n_features, n_actions, temp=1.):
        self.rng = rng
        self.weights = np.zeros((n_states, n_actions)) # we assume that n_features and n_actions for all agents are same
        self.temp = temp

    def value(self, phi, agent_action=None): #Phi is a n_features x numAgents dimensional matrix, joint_action is a list 
        if agent_action is None:
            v = np.sum(self.weights[phi,:], axis=0)
        v = np.sum(self.weights[phi, agent_action], axis=0)
        return v


    def pmf(self, phi):
        v = self.value(agent_state)/self.temp
        pmf = np.exp(v - logsumexp(v))
        return pmf


    def sample(self, phi):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))
    
    
class EgreedyPolicy:
    def __init__(self, rng, n_features, n_actions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((n_features, n_actions))

    def value(self, phi, agent_action=None):
        if agent_action is None:
            v = np.sum(self.weights[phi, :], axis=0)
        v = np.sum(self.weights[phi, agent_action], axis=0)
        return v

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            agent_action = int(self.rng.randint(self.weights.shape[1]))
        agent_action = int(np.argmax(self.value(phi))) 
        return agent_action

    
    #____________________________ Main algo _______________________
    
    
    
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
        parser.add_argument('--n_samples', help="Number of observation samples to compute mean vector and covariance matrix of Gaussian common belief", type=int, default=1000)
        # parser.add_argument('--n_mean_prior', help="Number of samples to compute mean vector of Gaussian common belief", type=int, default=100)
        # parser.add_argument('--n_cov_prior', help="Number of samples to compute covariance matrix of Gaussian common belief", type=int, default=100)
        parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
        
        
        args = parser.parse_args()

        rng = np.random.RandomState(1234)
        env = gym.make('FourroomsEnv-v0')
        
        f_name = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
        f_name = 'commonBeliefGaussian-mcReturn-fourroomsEnv-' + f_name + '.npy'
        
        possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]

        
        #traces = np.zeros((args.n_runs, args.n_episodes, args.n_agents+1))
        traces = {k:(0,np.zeros([env.observation_space.n,env.action_space.n])) for k in range(args.n_runs)}
        
        for run in range(args.n_runs):
            features = Tabular(env.observation_space.n)
            n_features, n_actions = len(features), env.action_space.n #len(features) = 1
            
            #common_belief_mean, common_belief_cov = MCR_Gauss(env, args.n_agents, maxIter).priors()
            common_belief_mean_posterior, common_belief_cov_posterior = Belief(self.n_agents).updateBeliefParameters(self, common_observation_samples)

            pi_policy = [SoftmaxPolicy(rng, n_features, args.n_actions) for _ in range(args.n_agents)]
            
            
            broadcasts = [Option().broadcast for _ in range(args.n_agents)]   
        
                
            #Compute the Monte Carlo return of episodes
            Q = MCR_Gauss(env, args.n_agents, args.n_mean_prior, args.n_cov_prior, args.n_steps).mc_episode(mean_prior, cov_prior, features(env.reset()), SoftmaxPolicy(rng, n_features, n_actions), args.n_episodes, args.n_steps)
            traces[run] = (step, Q)    
                    
            print('Run {} episode {} steps {} Q {}'.format(run, episode, step, Q))
        np.save(f_name, traces)
        print(f_name)