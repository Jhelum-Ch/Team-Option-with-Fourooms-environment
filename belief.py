import numpy as np
import scipy.stats

class Belief:

	#
	'''
	Implemented as in https://en.wikipedia.org/wiki/Conjugate_prior
	Help : https://stats.stackexchange.com/questions/312802/how-to-set-the-priors-for-bayesian-estimation-of-multivariate-normal-distributio
	'''
	def __init__(self, n_agents, sample_count=1000):

		# super(multivariateNormalBelief, self).__init__()
		#D : dimension
		self.D = n_agents

		'''
		Priors
		Query: Is it a good idea to set the dimension = number of agents?
		'''
		self.mu0 = np.zeros(self.D) #TODO: can be initialed uniformly randomly
		self.cov0 = np.eye(self.D) #TODO: can be initialed uniformly randomly

		#k_0 (conflict of definition with the wikipedia page)
		self.k0 = 0.1
		self.v0 = self.D + 1.5
		self.psi = (self.v0 - self.D - 1) * np.identity(self.D)

		#Number of samples
		self.N = sample_count

		self.num_itr = 100

		self.mean_itr = np.random.uniform(0, 1, self.D)
		self.cov_itr = scipy.stats.invwishart.rvs(self.v0, self.psi)

	def sample(self):
		'''
		purpose : samples observation from the current belief distribution
		returns : data matrix of dimenson (number of agents x sample_Count)
		'''
		samples = np.random.multivariate_normal(mean=self.mean_itr, cov=self.cov_itr, siez=self.N)
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
		x_bar = np.mean(samples,axis = 1) #sample mean
		sample_cov = np.cov(samples)

		#Gibb's sampling
		k = self.k0 + self.N
		v = self.v0 + self.N

		for _ in range(self.num_itr):
			#Update mean
			mean_tmp = (self.k0 * self.mu0 + self.N * x_bar) / (self.k0 + self.N)
			self.mean_itr = np.random.multivariate_normal(mean_tmp, self.cov_itr/k, 1)

			#Update cov
			sample_demean = samples - self.mean_itr
			C = np.dot((samples - self.mean_itr).T, (samples - self.mean_itr))
			scale_tmp = self.psi + C + (self.k0 * self.N) / (self.k0 + self.N) * np.dot((x_bar - self.mu0).T, (x_bar - self.mu0))
			self.cov_itr = scipy.stats.invwishart.rvs(v, scale_tmp)