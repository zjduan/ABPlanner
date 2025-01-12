import numpy as np
import torch
import torch.nn as nn
from torch.distributions.log_normal import LogNormal, Normal
from IPython import embed

class GRUActorCritic(nn.Module):

    def __init__(self,
                 n_stage,
                 obs_matrix_dim,
                 action_dim,
                 device,
                 hid_size=64,
                 use_rnn=True,
                 rnn_num_layers=1,
                 rnn_hid_size=128,
                 compute_std=True):
        super().__init__()

        self.n_stage = n_stage
        self.obs_matrix_dim = obs_matrix_dim
        self.action_dim = action_dim
        self.device = device

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_matrix_dim[0] * obs_matrix_dim[1] + 1, hid_size),
            nn.ReLU(),
        )

        cur_dim = hid_size
        self.rnn_hid_size = rnn_hid_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hid = None
        self.use_rnn = use_rnn
        if use_rnn is True:
            self.rnn = nn.GRU(cur_dim, rnn_hid_size, rnn_num_layers, batch_first=True) 
            cur_dim = rnn_hid_size

        self.compute_std = compute_std
        self.policy_mean = nn.Sequential(
            nn.Linear(cur_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, action_dim),
            # nn.Tanh()
        )
        if compute_std:
            self.log_std = nn.Parameter(np.log(0.1) * torch.ones(action_dim), requires_grad=True)

        self.value = nn.Sequential(
            nn.Linear(cur_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1)
        )
        self.use_LogNormal = False
        self.mean_scaler = 0.1
        self.std_scaler = 1.0

    def reset(self):
        self.rnn_hid = None

    def step(self, obs_matrix, budget):
        obs_matrix = torch.tensor(obs_matrix, dtype=torch.float32).unsqueeze(0) # (1, C, S)
        budget = torch.tensor(budget, dtype=torch.float32).view(1, 1)
        x = torch.concat([obs_matrix.view(1, -1), budget], -1).to(self.device)
        # print('inp', obs_matrix, budget, last_action)
        x = self.obs_encoder(x) # (1, d)
        # print('enc', x)
        if self.use_rnn:
            x = x.unsqueeze(1)          # (1, 1, d)
            if self.rnn_hid is not None:
                x, self.rnn_hid = self.rnn(x, self.rnn_hid)
            else:
                x, self.rnn_hid = self.rnn(x)
            x = x.squeeze(1)
        mean = self.policy_mean(x).view(-1) * self.mean_scaler
        if self.compute_std:
            std = self.std_scaler * torch.exp(self.log_std)
        else:
            std = self.std_scaler
        if self.use_LogNormal:
            policy = LogNormal(mean, std)
        else:
            policy = Normal(mean, std)

        value = self.value(x).view(-1)
        return policy, value

    def forward(self, observations, budgets):
        n_epi, epi_len = observations.shape[0:2]
        x = torch.concat([observations.view(n_epi, epi_len, -1),
                          budgets.view(n_epi, epi_len, 1)], -1)
        x = self.obs_encoder(x)
        if self.use_rnn:
            x, h = self.rnn(x)
            x = x.view(n_epi, epi_len, -1)          # (n_epi, epi_len, hidden_size)

        mean = self.policy_mean(x) * self.mean_scaler
        if self.compute_std:
            std = self.std_scaler * torch.exp(self.log_std)
        else:
            std = self.std_scaler
        if self.use_LogNormal:
            policy = LogNormal(mean, std)
        else:
            policy = Normal(mean, std)

        value = self.value(x)
        value = value.view(n_epi, epi_len)  # (n_epi, epi_len)

        return policy, value
