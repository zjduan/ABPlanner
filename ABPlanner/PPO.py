import os
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
from IPython import embed
from network import GRUActorCritic

from utils import projection_simplex_sort, EPS

class PPOAgent(object):

    def __init__(self,
                 n_stage,
                 obs_matrix_dim,
                 action_dim,
                 n_epi_per_task,
                 device,
                 learning_rate=3e-4,
                 gamma=0.99,
                 vf_coef=0.5,
                 ent_coef=5e-4,
                 eps_clip=0.2,
                 manual_step_coef=None):

        self.n_stage = n_stage
        self.obs_matrix_dim = obs_matrix_dim    # (C, S)
        self.action_dim = action_dim if action_dim > 0 else n_stage
        self.epi_len = n_epi_per_task - 1
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.w_vf = vf_coef
        self.w_ent = ent_coef
        self.eps_clip = eps_clip
        self.manual_step_coef = manual_step_coef

        self.target_kl = 0.01
        self.mode = 'train'
        self.actor_critic = GRUActorCritic(self.n_stage,
                                           self.obs_matrix_dim,
                                           self.action_dim,
                                           self.device).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

    def get_save_dict(self):
        self.save_dict = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        return self.save_dict

    def load_save_dict(self, save_dict, strict=True):
        self.actor_critic.load_state_dict(save_dict['actor_critic_state_dict'],
                                          strict=strict)
        self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        # self.n_update = save_dict['n_update']

    def train(self):
        self.actor_critic.train()
        self.mode = 'train'

    def eval(self):
        self.actor_critic.eval()
        self.mode = 'eval'

    def reset(self):
        self.actor_critic.reset()

    def step(self, obs_matrix, budget, test=False):
        last_budget_stage = obs_matrix[0]
        last_cost_stage = obs_matrix[2]

        with torch.no_grad():
            policy, value = self.actor_critic.step(obs_matrix, budget)

        if test is not True:
            action = policy.sample().view(-1) # (action_dim)
        else:
            action = policy.mean

        logp = policy.log_prob(action).view(-1).cpu().numpy() # (action_dim)
        action = action.cpu().numpy()

        # budget_stage = last_budget_stage + action
        budget_stage = last_budget_stage + (action - action.mean())
        budget_stage = projection_simplex_sort(budget_stage, budget)

        info = {
            'action': action,
            'logp': logp,
            'value': value,
            # 'alpha': alpha
        }
        return budget_stage, info

        return budget_stage, info

    def learn(self, data, repeat=10):
        self.train()
        info = dict()

        observations, budgets, actions, rewards, _values, logps, advantages, returns = data
        n_epi, epi_len = observations.shape[0:2]
        observations = torch.tensor(observations, dtype=torch.float32).to(self.device)          # (n_epi, epi_len, C, S)
        budgets      = torch.tensor(budgets, dtype=torch.float32).to(self.device)               # (n_epi, epi_len)
        rewards      = torch.tensor(rewards, dtype=torch.float32).to(self.device)               # (n_epi, epi_len)
        returns      = torch.tensor(returns, dtype=torch.float32).to(self.device)               # (n_epi, epi_len)
        actions  = torch.tensor(actions, dtype=torch.float32).to(self.device)           # (n_epi, epi_len, action_dim)
        logps        = torch.tensor(logps, dtype=torch.float32).to(self.device)                 # (n_epi，epi_len, action_dim)
        advantages   = torch.tensor(advantages, dtype=torch.float32).to(self.device)            # (n_epi, epi_len,)

        # print(observations.shape, actions.shape, logps.shape, advantages.shape)
        for i in range(repeat):
            policy, new_values = self.actor_critic(observations, budgets)

            new_logps = policy.log_prob(actions)

            ratio = torch.exp((new_logps - logps).sum(-1))                  # (n_epi, epi_len)
            clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_pi = -1.0 * torch.minimum(ratio * advantages, clip_adv).mean()
            approx_kl = (logps - new_logps).sum(-1).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                logging.info(f'Early terminate at {i}')
                break
            ent = policy.entropy().mean()

            target_v = rewards.clone()
            target_v[:, :-1] += self.gamma * (new_values.detach())[:, 1:]
            loss_vf = ((new_values - target_v)**2).mean()
            # loss_vf = ((new_values - returns)**2).mean()

            loss = loss_pi - self.w_ent * ent + self.w_vf * loss_vf

            info['LossPi'] = float(loss_pi)
            info['Ent'] = ent.item()
            info['LossVF'] = loss_vf.item()


            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
            self.optimizer.step()

        return info
