import numpy as np
import random
import copy

class OU_Noise(object):

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state


class epsilon(object) :
    def __init__(self, max_step,epsilon_max,epsilon_min):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.max_step = max_step

        self.eps = epsilon_min

    """
    전체 스텝의 ratio 만큼까지 선형적으로 증가
    step은 현재 step 진척도
    """

    def linear(self,step,ratio=0.75) :
        max_eps_step = self.max_step*ratio

        if step <= max_eps_step :
            self.eps = self.epsilon_max - (self.epsilon_max - self.epsilon_min)*step/max_eps_step
        
        else :
            self.eps = self.epsilon_min

        return self.eps
    def proportional(self, step,ratio=0.75) :
        max_eps_step = self.max_step*ratio

        if step <= max_eps_step :
            self.eps = max((self.epsilon_max*max_eps_step/10/(step + max_eps_step/10)), self.epsilon_min)
        
        else :
            self.eps = self.epsilon_min

        return self.eps        
