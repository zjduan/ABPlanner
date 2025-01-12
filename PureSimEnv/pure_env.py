import os
import logging
import numpy as np
from copy import deepcopy
from IPython import embed
import matplotlib.pyplot as plt

from .spa_env import SPAEnv

class BudgetAllocationEnv(object):
    def __init__(self,
                 n_stage,
                 auction_episode_len=10000,
                 fix_budget=True,
                 fix_value=False,
                 n_obs_row=5,
                 agent=None,
                 data_dir=None):
        self.n_stage = n_stage
        self.auction_episode_len = auction_episode_len
        self.fix_budget = fix_budget
        self.fix_value = fix_value

        self.auction_env = SPAEnv(epi_len=auction_episode_len)
        # self.scaler = n_stage / auction_episode_len if scaler is None else scaler
        self.n_obs_row = n_obs_row

        if agent is not None:
            self.agent_name = agent.name
            self.low_agent = agent
        else:
            self.agent_name = 'Greedy'
            self.low_agent = None


        self.data_dir = data_dir
        self.phase = 'train'

    def train(self):
        self.phase = 'train'

    def test(self):
        self.phase = 'test'
        data_dir = self.data_dir
        if data_dir is not None:
            logging.info(f'Load data from {data_dir}')
            self.budget_data = np.load(os.path.join(data_dir, 'budget.npy'))
            self.feature_data = np.load(os.path.join(data_dir, 'feature.npy'))
            self.t_starts_data = np.load(os.path.join(data_dir, 't_starts.npy'))
            self.n_campaign = len(self.budget_data)
            self.n_task = self.n_campaign
            self.data_idx = 0

    def reset(self):
        if self.phase == 'test' and self.data_dir is not None:
            self.budget = self.budget_data[self.data_idx]
            self.feature = self.feature_data[self.data_idx]
            self.t_starts = self.t_starts_data[self.data_idx]
            self.data_idx += 1
            obs = self.auction_env.reset(self.feature, sample_budget=False)
        else:
            obs = self.auction_env.reset()
            self.budget = obs['budget']
            self.feature = obs['feature']

            division = np.ones(self.n_stage)
            division = division * np.random.lognormal(0, 0.5, self.n_stage)
            division = division / division.sum()

            self.division = division
            self.t_starts = np.zeros(self.n_stage + 1, dtype=np.int32)
            self.t_starts[0] = 0
            for i in range(self.n_stage):
                self.t_starts[i+1] = self.t_starts[i] + int(division[i] * self.auction_episode_len)
            self.t_starts[-1] = self.auction_episode_len

        self.scaler = self.n_stage / self.budget
        return self.feature, self.budget * self.scaler

    def get_init_budget_stage(self, scaled_budget, init_method=None):
        if init_method == 'average':
            return scaled_budget / self.n_stage * np.ones(self.n_stage)
        elif init_method == 'random':
            x = np.random.rand(self.n_stage)
            return scaled_budget * x / x.sum()
        else:
            return 0

    def get_stage_result(self, scaled_budget_stage):
        budget_stage = scaled_budget_stage / self.scaler

        result = self._get_stage_result(budget_stage)
        return_stage, cost_stage, margin_ROI, margin_cost = result

        return [return_stage*self.scaler, cost_stage*self.scaler, margin_ROI]

    def get_singe_stage_result(self, stage_id, scaled_allocated_budget):
        allocated_budget = scaled_allocated_budget / self.scaler

        result = self._get_singe_stage_result(stage_id, allocated_budget)
        return np.array(result) * self.scaler

    def _get_singe_stage_result(self, stage_id, allocated_budget, consumed_budget=0):
        i = stage_id
        if self.agent_name == 'PID':
            result, info = self.auction_env.get_stage_result(self.low_agent, allocated_budget,
                                                             self.t_starts[i], self.t_starts[i+1])
        elif 'USCB' in self.agent_name:
            consumed_budget = consumed_budget
            remain_budget = allocated_budget
            USCB_remain_budget = allocated_budget

            result, info = self.auction_env.get_stage_result(self.low_agent, remain_budget,
                                                             self.t_starts[i], self.t_starts[i+1],
                                                             consumed_budget=consumed_budget,
                                                             USCB_remain_budget=USCB_remain_budget)
        else:
            result, info = self.auction_env.get_greedy_solution(allocated_budget,
                                                                self.t_starts[i], self.t_starts[i+1])

        return result

    def _get_stage_result(self, budget_stage):
        if not isinstance(budget_stage, np.ndarray):
            assert 'USCB' in self.agent_name or 'PID' in self.agent_name
            acc_rewards, acc_payments = self.auction_env.get_stage_result(self.low_agent, self.budget,
                                                                            t_starts=self.t_starts)
            return [acc_rewards, acc_payments, np.zeros(self.n_stage), np.zeros(self.n_stage)]

        return_stage, cost_stage, margin_ROI, margin_cost \
            = np.ones(self.n_stage), np.ones(self.n_stage), np.ones(self.n_stage), np.ones(self.n_stage)

        for i in range(self.n_stage):
            result = self._get_singe_stage_result(i, budget_stage[i], consumed_budget=cost_stage[:i].sum())
            return_stage[i], cost_stage[i], margin_ROI[i], margin_cost[i] = result

        return [return_stage, cost_stage, margin_ROI, margin_cost]

    def step(self, scaled_budget_stage):
        info = {}
        # assert sum(scaled_budget_stage) <= self.budget*self.scaler + 1e-3
        budget_stage = scaled_budget_stage / self.scaler

        result = self._get_stage_result(budget_stage)
        return_stage, cost_stage, margin_ROI, margin_cost = result
        if not isinstance(budget_stage, np.ndarray):
            budget_stage = budget_stage * np.zeros(self.n_stage)

        self.set_new_episode()

        if self.n_obs_row == 3:
            next_obs = np.stack([budget_stage*self.scaler, return_stage*self.scaler, cost_stage*self.scaler],
                                0)
        else:
            next_obs = np.stack([budget_stage*self.scaler, return_stage*self.scaler, cost_stage*self.scaler,
                                 margin_ROI, margin_cost*self.scaler], 0)

        info["margin_ROI"] = margin_ROI

        return next_obs, self.budget*self.scaler, info

    def set_new_episode(self):
        self.auction_env.set_new_episode(self.fix_value)

    def get_random_result(self, scaled_budget):
        budget = scaled_budget / self.scaler
        logits = np.random.rand(self.n_stage)
        budget_stage = budget * logits / logits.sum()

        result = self._get_stage_result(budget_stage)
        return_stage, cost_stage, margin_ROI, margin_cost = result

        return return_stage.sum()*self.scaler

    def get_average_result(self, scaled_budget):
        budget = scaled_budget / self.scaler
        budget_stage = budget / self.n_stage * np.ones(self.n_stage)

        result = self._get_stage_result(budget_stage)
        return_stage, cost_stage, margin_ROI, margin_cost = result

        return return_stage.sum()*self.scaler


    def get_greedy_result(self, scaled_budget):
        budget = scaled_budget / self.scaler

        result, _ = self.auction_env.get_greedy_solution(budget)
        acc_reward, acc_payment, margin_ROI, _ = result
        return acc_reward  * self.scaler

    def get_PID_result(self, scaled_budget):
        budget = scaled_budget / self.scaler

        result, info = self.auction_env.get_stage_result(self.low_agent, budget)
        acc_reward, acc_payment, __, _ = result

        return acc_reward  * self.scaler
