import numpy as np
from utils import discount_cumsum

class ReplayBuffer(object):

    def __init__(self, n_task, n_epi_per_task, n_stage, action_dim, gamma=0.99, lamb=0.95):
        self.n_epi   = n_task
        self.epi_len = n_epi_per_task - 1
        self.n_stage = n_stage
        self.action_dim = action_dim
        self.gamma   = gamma
        self.lamb    = lamb
        self.return_buf     = np.zeros((self.n_epi, self.epi_len), dtype=np.float32)
        self.advantage_buf  = np.zeros((self.n_epi, self.epi_len), dtype=np.float32)

    def fit(self, obs_list, budget_list, action_list, reward_list, value_list, logp_list):
        self.observation_buf = np.array(obs_list).reshape((self.n_epi, self.epi_len, *obs_list[0].shape))
        self.budget_buf = np.array(budget_list).reshape((self.n_epi, self.epi_len))
        self.action_buf = np.array(action_list).reshape((self.n_epi, self.epi_len, self.action_dim))
        self.reward_buf = np.array(reward_list).reshape((self.n_epi, self.epi_len))
        self.value_buf = np.array(value_list).reshape((self.n_epi, self.epi_len))
        self.logp_buf = np.array(logp_list).reshape((self.n_epi, self.epi_len, self.action_dim))

        deltas = self.reward_buf - self.value_buf
        deltas[:, :-1] += self.gamma * self.value_buf[:, 1:]
        for i in range(self.n_epi):
            self.advantage_buf[i] = discount_cumsum(deltas[i], self.gamma * self.lamb)

        for i in range(self.n_epi):
            self.return_buf[i] = discount_cumsum(self.reward_buf[i], self.gamma)
    
    def get(self):
        mean, std = np.mean(self.advantage_buf), np.std(self.advantage_buf)
        advantage_buf = (self.advantage_buf - mean) / std

        return self.observation_buf, self.budget_buf, self.action_buf, self.reward_buf, \
            self.value_buf, self.logp_buf, advantage_buf, self.return_buf