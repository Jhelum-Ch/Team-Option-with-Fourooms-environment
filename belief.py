import numpy as np
import scipy.stats
import random


class Belief:
    #
    '''
    Implemented as in https://en.wikipedia.org/wiki/Conjugate_prior
    Help : https://stats.stackexchange.com/questions/312802/how-to-set-the-priors-for-bayesian-estimation-of-multivariate-normal-distributio
    '''

    def __init__(self, n_agents, states_list, sample_count=1000):

        # super(multivariateNormalBelief, self).__init__()
        # D : dimension
        self.D = n_agents
        self.states_list = states_list

        '''
        Priors
        Query: Is it a good idea to set the dimension = number of agents?
        '''
        self.mu0 = np.zeros(self.D)  # TODO: can be initialed uniformly randomly
        self.cov0 = np.eye(self.D)  # TODO: can be initialed uniformly randomly

        # k_0 (conflict of definition with the wikipedia page)
        self.k0 = 0  # 0.1
        self.v0 = self.D + 2  # self.D + 1.5
        assert isinstance(self.k0, int) and isinstance(self.v0, int) == True, 'k0 and v0 must be integers'

        self.psi = (self.v0 - self.D - 1) * np.identity(self.D)

        # Number of samples
        self.N = sample_count

        self.num_itr = 100

        # self.mean_itr = np.random.uniform(0, 1, self.D)
        self.mean_itr = random.sample(self.states_list, k=self.D)
        self.cov_itr = scipy.stats.invwishart.rvs(self.v0, self.psi)

    def sample(self):
        '''
        purpose : samples observation from the current belief distribution
        returns : data matrix of dimenson (number of agents x sample_Count)
        '''
        # TODO: Ensure samples are composed of discrete atomic states (i.e. Integers, not decimal numbers)
        samples = np.random.multivariate_normal(mean=self.mean_itr, cov=self.cov_itr, size=self.N)
        return samples

    def sample_single_state(self):
        '''
        purpose : sample a unique state from the current belief distribution
        returns : joint state tuple
        '''

        # TODO: Ensure samples are composed of discrete atomic states (i.e. Integers, not decimal numbers)
        sample = np.random.multivariate_normal(mean=self.mean_itr, cov=self.cov_itr, size=1)

        state_list = []
        for i in range(sample.size):
            state_list.append(round(sample[0][i]))      # TODO: Patched temporarily with round()

        return tuple(state_list)

    def updateBeliefParameters(self, samples):
        '''
        uses Normal Inverse Wishart for posterior update of the parameters of the prior distribution
        '''

        x_bar = np.mean(samples, axis=0)  # sample mean
        sample_cov = np.cov(samples)

        # Gibb's sampling
        k = self.k0 + self.N
        v = self.v0 + self.N

        for _ in range(self.num_itr):
            # Update mean
            mean_tmp = (self.k0 * self.mu0 + self.N * x_bar) / (self.k0 + self.N)
            # print(mean_tmp)
            self.mean_itr = np.random.multivariate_normal(mean_tmp, self.cov_itr / k)

            # Update cov
            sample_demean = samples - self.mean_itr
            C = np.dot((samples - self.mean_itr).T, (samples - self.mean_itr))
            scale_tmp = self.psi + C + (self.k0 * self.N) / (self.k0 + self.N) * np.dot((x_bar - self.mu0).T,
                                                                                        (x_bar - self.mu0))
            self.cov_itr = scipy.stats.invwishart.rvs(v, scale_tmp)
