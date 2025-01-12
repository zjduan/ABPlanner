import numpy as np
from copy import deepcopy
from utils import discount_cumsum

class ReplayBuffer(object):

    def __init__(self, buffer_size, dim_state, dim_action):
        self.buffer_size = buffer_size
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.state_buf = np.zeros((buffer_size, self.dim_state))
        self.action_buf = np.zeros((buffer_size, self.dim_action))
        self.reward_buf = np.zeros((buffer_size, 1))
        self.next_state_buf = np.zeros((buffer_size, self.dim_state))
        self.done_buf = np.zeros((buffer_size, 1))
        self.iter = 0
        self.cur_size = 0

    def store(self, state, action, reward, next_state, done):
        self.state_buf[self.iter] = state
        self.action_buf[self.iter] = action
        self.reward_buf[self.iter] = reward
        self.next_state_buf[self.iter] = next_state
        self.done_buf[self.iter] = done

        self.iter += 1
        if self.iter == self.buffer_size:
            self.is_full = False
            self.iter = 0
        if self.cur_size < self.buffer_size:
            self.cur_size += 1

    def sample(self, batch_size):
        if self.cur_size < batch_size:
            return None
        indices = np.random.choice(self.cur_size, batch_size)

        data = dict()
        data["states"] = self.state_buf[indices]
        data["actions"] = self.action_buf[indices]
        data["rewards"] = self.reward_buf[indices]
        data["next_states"] = self.next_state_buf[indices]
        data["dones"] = self.done_buf[indices]

        return data
