import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.log_normal import LogNormal, Normal

from IPython import embed
class Critic(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        self.net = nn.Sequential(
            nn.Linear(self.dim_observation + dim_action, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        combined = torch.cat([obs, acts], -1)
        result = self.net(combined)

        return result

class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_observation, 32),
            nn.ReLU(),
            nn.Linear(32, dim_action)
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(dim_action), requires_grad=True)

    def forward(self, obs):
        result = self.net(obs)
        log_means = torch.tanh(result) * 2

        std = torch.exp(self.log_std)
        means = torch.exp(log_means)

        policies = LogNormal(log_means, std)
        return policies, means
