import os
import sys

import math
import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy

from PureSimEnv.USCBAgent.replay_buffer import ReplayBuffer
from PureSimEnv.USCBAgent.network import Actor, Critic

from IPython import embed

class USCBAgent:
    def __init__(self,
                 dim_state=3,
                 dim_action=1,
                 gamma=1.0,
                 tau=0.05,
                 critic_lr=3e-4,
                 actor_lr=3e-4,
                 buffer_size=500000,
                 sample_size=64,
                 device='cpu',
                 act_freq=50):

        self.name = "USCB"
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        # actor and critic and their targets
        self.actor = Actor(self.dim_state, self.dim_action)

        self.critic = Critic(self.dim_state, self.dim_action)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.gamma = gamma
        self.tau = tau
        self.num_of_steps = 0

        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)

        # cuda usage
        self.device = device
        self.use_cuda = 'cuda' in self.device
        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        # replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, dim_state, dim_action)
        self.buffer_pointer = 0
        self.buffer_size = buffer_size
        self.sample_size = sample_size

        self.act_freq = act_freq

    def update_cur_state(self):
        self.cur_state = np.zeros(self.dim_state)
        self.cur_state[0] = self.cur_step / 100.0
        self.cur_state[1] = self.remain_step / 100.0
        self.cur_state[2] = self.remain_budget / 100.0

    def reset(self, remain_T, remain_budget, cur_T = 0, consumed_budget = 0):
        self.t = 0

        self.cur_step = math.ceil(cur_T / self.act_freq)
        self.remain_step = math.ceil(remain_T / self.act_freq)
        self.consumed_budget = consumed_budget
        self.remain_budget = remain_budget

        self.update_cur_state()

        self.cur_action = None
        self.cur_reward = 0
        self.cur_payment = 0

    def train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def step(self, mode="train"):
        if self.t % self.act_freq == 0:
            states = torch.tensor(self.cur_state, dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                policies, means = self.actor(states)
            if mode == "train":
                actions = policies.sample()
            else:
                actions = means
            action = actions.view(-1).cpu().numpy()
            self.cur_action = action

        return self.cur_action

    def update(self, reward, payment, done=False, mode="eval"):
        self.cur_reward += reward
        self.cur_payment += payment
        self.t += 1
        if self.t % self.act_freq == 0 or done == True:
            reward = self.cur_reward
            payment = self.cur_payment
            self.cur_step += 1
            self.remain_step -= 1
            self.consumed_budget += payment
            self.remain_budget -= payment

            if mode == 'train':
                state = deepcopy(self.cur_state)
                action = deepcopy(self.cur_action)

            self.update_cur_state()
            if mode == 'train':
                next_state = deepcopy(self.cur_state)
                # print(state, action, reward, next_state, done)
                self.replay_buffer.store(state, action, reward / 100.0, next_state, done)
                self.learn()

            self.cur_reward = 0
            self.cur_payment = 0


    def learn(self):
        if self.replay_buffer.cur_size < 10000 or self.replay_buffer.iter % 10 != 0:
            return

        data = self.replay_buffer.sample(self.sample_size)
        if data is None:
            return
        
        states = torch.tensor(data["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(data["actions"], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(data["next_states"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(data["dones"], dtype=torch.float32).to(self.device)

        for i in range(1):
            current_Q = self.critic(states, actions)
            with torch.no_grad():
                next_policies, next_means = self.actor_target(next_states)
                next_actions = next_means
                target_Q = rewards + self.gamma * self.critic_target(next_states, next_actions) * (1 - dones)

            loss_Q = ((current_Q - target_Q)**2).mean()
            self.critic_optimizer.zero_grad()
            loss_Q.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
            self.critic_optimizer.step()

        for i in range(1):
            policies_this_agent, _ = self.actor(states)
            actions_this_agent = policies_this_agent.rsample()
            loss_A = -self.critic(states, actions_this_agent)
            loss_A = loss_A.mean()

            self.actor_optimizer.zero_grad()
            loss_A.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
            self.actor_optimizer.step()

        for target_param, source_param in zip(self.critic_target.parameters(),
                                              self.critic.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * source_param.data)
        for target_param, source_param in zip(self.actor_target.parameters(),
                                              self.actor.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * source_param.data)

    def get_save_dict(self):
        self.save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer
        }
        return self.save_dict

    def load_save_dict(self, save_dict, mode='train'):
        self.actor.load_state_dict(save_dict['actor_state_dict'])
        if mode == 'train':
            self.critic.load_state_dict(save_dict['critic_state_dict'])
            self.actor_target.load_state_dict(save_dict['actor_target_state_dict'])
            self.critic_target.load_state_dict(save_dict['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(save_dict['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(save_dict['critic_optimizer_state_dict'])
            self.replay_buffer = save_dict['replay_buffer']
