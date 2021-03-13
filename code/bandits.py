import numpy as np

class BernoulliBandit(object):

    def __init__(self, K, probs=None, seed=21):
        
        if probs is None:
            # generate random paramaters
            np.random.seed(seed)
            self.probs = [np.random.random() for _ in range(K)]
        else:
            self.probs = probs

        self.K = K
        self.best_prob = max(self.probs)

    def get_reward(self, i):
        if np.random.random() < self.probs[i]:
            return 1
        else:
            return 0