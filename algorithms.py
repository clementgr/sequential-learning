import numpy as np

from utils import kl_bernoulli, bisection


class DefaultAlgorithm(object):
    
    def __init__(self, bandit):

        self.bandit = bandit
        self.step = 0
        self.mean_estimates = [1.0] * self.bandit.K
        self.var_estimates = [1.0] * self.bandit.K
        self.upper_bounds = [1.0] * self.bandit.K
        self.counts = [0] * self.bandit.K
        self.reward = 0
        self.rewards = [0] * self.bandit.K
        self.regret = 0
        self.cum_regret = []

    def mean_estimates(self):
        return self.mean_estimates
    
    def var_estimates(self):
        return self.var_estimates
    
    def reset(self):
        self.step = 0
        self.mean_estimates = [1.0] * self.bandit.K
        self.var_estimates = [1.0] * self.bandit.K
        self.upper_bounds = [1.0] * self.bandit.K
        self.counts = [0] * self.bandit.K
        self.reward = 0
        self.rewards = [0] * self.bandit.K
        self.regret = 0
        self.cum_regret = []

    def update(self, i):
        # update count
        self.counts[i] += 1
        # update reward
        self.reward = self.bandit.get_reward(i)
        self.rewards[i] += self.reward
        # update estimate
        self.mean_estimates[i] += 1 / self.counts[i] * (self.reward - self.mean_estimates[i])
        # update regret
        self.regret += self.bandit.best_prob - self.bandit.probs[i]
        self.cum_regret.append(self.regret)
  
    def select_next_arm(self):
        raise NotImplementedError

    def run(self, n_step, n_run=None, ntot_run=None):
        for j in range(n_step):
            if n_run is not None:
                print(f'\rrun {n_run+1}/{ntot_run} | running step {j+1}/{n_step}', end='')
            else:
                print(f'\rrunning step {j+1}/{n_step}', end='')
            i = self.select_next_arm()
            self.update(i)


class FTL(DefaultAlgorithm):
  
    def select_next_arm(self):
        self.step += 1
        for k in range(self.bandit.K):
            if self.counts[k] == 0:
                return k
        return np.argmax(self.mean_estimates)

class UCB(DefaultAlgorithm):
    
    def __init__(self, bandit, sigma_square):
        super(UCB, self).__init__(bandit)
        self.sigma_square = sigma_square
        
    def select_next_arm(self):
        self.step += 1
        for k in range(self.bandit.K):
            if self.counts[k] == 0:
                return k
            sqrt_quantity = np.sqrt(2 * np.log(self.step) * self.sigma_square / self.counts[k])
            self.upper_bounds[k] = self.mean_estimates[k] + sqrt_quantity
        return np.argmax(self.upper_bounds)


class UCB_V(DefaultAlgorithm):
    
    def __init__(self, bandit, b, ksi, c):
        super(UCB_V, self).__init__(bandit)
        self.b = b
        self.ksi = ksi
        self.c = c
    
    def update(self, i):
        # update count
        previous_count = self.counts[i]
        self.counts[i] += 1
        # update reward
        self.reward = self.bandit.get_reward(i)
        self.rewards[i] += self.reward
        # update estimate
        previous_mean_estimate = self.mean_estimates[i]
        self.mean_estimates[i] += 1 / self.counts[i] * (self.reward - self.mean_estimates[i])
        mean_quantity = (self.reward - previous_mean_estimate) * (self.reward - self.mean_estimates[i])
        self.var_estimates[i] = (previous_count * self.var_estimates[i] + mean_quantity) / self.counts[i]
        # update regret
        self.regret += self.bandit.best_prob - self.bandit.probs[i]
        self.cum_regret.append(self.regret)
    
    def select_next_arm(self):
        self.step += 1
        for k in range(self.bandit.K):
            if self.counts[k] == 0:
                return k
            sqrt_quantity = np.sqrt(2 * np.log(self.step) * self.var_estimates[k] * self.ksi / self.counts[k])
            self.upper_bounds[k] = self.mean_estimates[k] + sqrt_quantity + 3*self.b*self.c*self.ksi / self.counts[k]
        return np.argmax(self.upper_bounds)


class kl_UCB(DefaultAlgorithm):
  
    def select_next_arm(self):
        self.step += 1
        for k in range(self.bandit.K):
            if self.counts[k] == 0:
                return k
            else:
                self.upper_bounds[k] = bisection(self.rewards, self.counts, k, self.step)
        return np.argmax(self.upper_bounds)