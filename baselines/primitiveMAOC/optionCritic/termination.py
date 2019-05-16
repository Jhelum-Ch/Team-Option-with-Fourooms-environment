import numpy as np
from scipy.special import expit
from modelConfig import seed


class SigmoidTermination:
    def __init__(self, n_states):
        self.rng = seed
        self.weights = np.zeros(n_states)

    def pmf(self, phi):
        pmf = expit(np.sum(self.weights[phi]))
        return pmf

    def sample(self, phi):
        sample_terminate = int(self.rng.uniform() < self.pmf(phi))
        return sample_terminate

    def grad(self, phi): # Check this formula
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi


class OneStepTermination:
    def sample(self, phi):
        return 1
        #return [1 for _ in range(np.shape(phi)[1])]

    def pmf(self, phi):
        return 1.
        #return [1. for _ in range(np.shape(phi)[1])]
