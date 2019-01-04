import numpy as np
import scipy.stats
import random


class BeliefGaussian:
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
        samples = np.random.multivariate_normal(mean=self.mean_itr, cov=self.cov_itr, size=self.N)
        return samples

    def sample_single_state(self):
        '''
        purpose : sample a unique state from the current belief distribution
        returns : joint state tuple
        '''
        sample = np.random.multivariate_normal(mean=self.mean_itr, cov=self.cov_itr, size=1)

        state_list = []
        for i in range(sample.size):
            state_list.append(sample[0][i])

        return tuple(state_list)

    def updateBeliefParameters(self, samples):
        '''
        uses Normal Inverse Wishart for posterior update of the parameters of the prior distribution
        '''
        x_bar = np.mean(samples, axis=0)  # sample mean
        print(x_bar.size)
        sample_cov = np.cov(samples)

        # Gibb's sampling
        k = self.k0 + self.N
        v = self.v0 + self.N

        for _ in range(self.num_itr):
            # Update mean
            mean_tmp = (self.k0 * self.mu0 + self.N * x_bar) / (self.k0 + self.N)
            print(mean_tmp)
            self.mean_itr = np.random.multivariate_normal(mean_tmp, self.cov_itr / k, 1)

            # Update cov
            sample_demean = samples - self.mean_itr
            C = np.dot((samples - self.mean_itr).T, (samples - self.mean_itr))
            scale_tmp = self.psi + C + (self.k0 * self.N) / (self.k0 + self.N) * np.dot((x_bar - self.mu0).T,
                                                                                        (x_bar - self.mu0))
            self.cov_itr = scipy.stats.invwishart.rvs(v, scale_tmp)



class TruncatedNormal:

    def __init__(self, mu, sigma, a, b):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.num_params = 1000
        self.x = np.linspace(self.a, self.b, self.num_params)
        
        
    def pdf(self):
        deno = self.sigma*(stats.norm.cdf(self.b, loc=self.mu, scale=self.sigma) - stats.norm.cdf(self.a, loc=self.mu, scale=self.sigma))
        nume = [stats.norm.pdf(self.x[i], loc=self.mu, scale=self.sigma) for i in range(len(self.x))]
        
        return (nume/deno)/np.sum(nume/deno)
    
    def sample(self):
        return np.random.choice(self.x, p=self.pdf())    


# def rejSample(t, mu, sigma):
#     while True:
#         Y = t.sample()
#         X = np.rint(Y)
#         Z = np.abs(X) + 0.5
#         U = np.random.uniform(0,1)
#         if -2.*(sigma**2)*np.log(U) >= (Z-mu)**2 - (Y-mu)**2:
#             break
#     return X


class discreteTruncMultivariateGaussian():
    def __init__(self, mu_vec, sigma_mat, a, b):
        self.mu_vec = mu_vec
        self.sigma_mat = sigma_mat
        self.a = a
        self.b = b
        #super(discreteTruncMultivariateGaussian, self).__init__()
        
    def truncateNormalpdf(self,x):
        rv = scipy.stats.truncnorm(self.a, self.b)
        return rv.pdf(x)
    
    def truncateNormalsample(self, mu, sigma, size):
        return truncnorm.rvs(self.a, self.b, loc = mu, scale = sigma, size=size, random_state=None) 
    
    def rejSample(self):
        X = np.zeros(len(self.mu_vec))
        Y = np.zeros_like(X)
        for i in range(len(self.mu_vec)):
            if i==0:
                mu = self.mu_vec[i] 
                sigma = self.sigma_mat[i,i]  

            elif i==1:
                A = self.sigma_mat[i-1,i-1]
                A = float(A)

                a = self.sigma_mat[i-1,i]
                a = float(a)

                W = Y[i-1]
                W = float(W)

                mu = self.mu_vec[i] + a*(1./A)*W 
                sigma = self.sigma_mat[i,i] - a*(1./A)*a
                
            else:
                A = self.sigma_mat[0:i-1,0:i-1]
                A = np.reshape(A, (i-1,i-1))

                a = self.sigma_mat[0:i-1,i]
                a = np.reshape(a, (i-1,1))

                W = Y[0:i-1]
                W = np.reshape(W, (i-1,1))

                mu = np.array(self.mu_vec[i] + np.matmul(a.T, np.matmul(np.linalg.inv(A),W)))
                sigma = np.array(self.sigma_mat[i,i] - np.matmul(a.T, np.matmul(np.linalg.inv(A),a)))


            flag = 0
            while flag==0:
                #Y = t.sample()
                y = self.truncateNormalsample(mu, sigma, 1)
                x = np.rint(y)
                z = np.abs(x) + 0.5
                u = np.random.uniform(0,1)
                if -2.*(sigma**2)*np.log(u) >= (z-mu)**2 - (y-mu)**2:
                    flag=1
            
            X[i] = int(x)
            Y[i] = y
        return X     



