import numpy as np


def bernoulli_var(p):
  return p * (1-p)


def sub_gaussian_sigma(p):
  if p == 0 or p == 1:
    return 0
  elif p == 1./2:
    return 1./4
  else:
    num = 2*p -1
    denom = np.log(p) - np.log(1-p)
    return num / denom / 2


def kl_bernoulli(p, q, epsilon=1e-10):
    if p < epsilon:
        if q > epsilon:
            return np.log(1/(1-q))
        else:
            return p * np.log(p/q)
    if q < epsilon:
        if p > epsilon:
            return 1e10
        else:
            return p * np.log(p/q)
    if 1-p < epsilon:
        if 1-q > epsilon:
            return np.log(1/q)
        else:
            return (1-p) * np.log((1-p)/(1-q))
    if 1-q < epsilon:
        if 1-p > epsilon:
            return 1e10
        else:
            return (1-p) * np.log((1-p)/(1-q))
    else:
        return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))


def bisection(rewards, counts, k, step, epsilon=1e-6, max_iter=1000):
  
    upper_bound = np.log(1 + step*np.log(step)**2) / counts[k]
    expected_reward = rewards[k] / counts[k]

    a = expected_reward
    b = 1
    n = 0

    while n < max_iter and b - a > epsilon:
        c = (a + b)/2
        if kl_bernoulli(expected_reward, c) < upper_bound:
            a = c
        else:
            b = c
        n += 1

    return (a+b)/2