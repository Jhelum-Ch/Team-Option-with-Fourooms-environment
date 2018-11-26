import numpy as np
from scipy.special import expit


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


class OneStepTermination:
    def sample(self, phi):
        return [1 for _ in range(np.shape(phi)[1])]

    def pmf(self, phi):
        return [1. for _ in range(np.shape(phi)[1])]