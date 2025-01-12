import os
import numpy as np
from copy import deepcopy
from IPython import embed
import matplotlib.pyplot as plt
from tqdm import tqdm

class SPAEnv(object):
    def __init__(self, epi_len=100):
        self.epi_len = epi_len
        self.t = None
        self.d_feature = 3
        self.values = None
        self.market_prices = None

    def reset(self, feature=None, sample_budget=True, epi_start=0):
        self.t = epi_start
        self.feature = feature if feature is not None else np.random.rand(self.d_feature)
        obs = self.set_new_episode(fix_value=False)
        obs['feature'] = self.feature
        if sample_budget == True:
            obs['budget'] = int(np.random.uniform(100, 200))
        return obs

    def set_epi_start(self, epi_start):
        self.t = epi_start

    def set_new_episode(self, fix_value=False):
        self.t = 0
        if (self.values is None) or fix_value == False:
            self.market_prices = np.random.lognormal(np.log(0.1), 0.1, self.epi_len)
            x = np.linspace(0, 2 * np.pi, self.epi_len) - self.feature[1]
            y = np.cos(x) * self.feature[2]
            ROI = np.random.pareto(3 + y, self.epi_len)
            self.values = self.market_prices * ROI

        obs = {
            "value": self.values[self.t],
        }
        return obs

    def replay_episode(self):
        self.t = 0
        obs = {
            "value": self.values[self.t],
        }
        return obs

    def get_stage_result(self, agent, budget, t_start=0, t_end=None,
                           consumed_budget = 0, t_starts=None, USCB_remain_budget=None):
        t_end = self.epi_len if t_end is None else t_end
        if 'USCB' in agent.name:
            if USCB_remain_budget is None:
                USCB_remain_budget = budget
            agent.reset(remain_T=t_end - t_start, remain_budget=USCB_remain_budget, cur_T=t_start, consumed_budget=0)
        else:
            agent.reset(budget, t_end - t_start)
        budget_stage = 0

        if t_starts is not None:
            stage = 0
            acc_rewards = np.zeros(len(t_starts) - 1)
            acc_payments = np.zeros(len(t_starts) - 1)

        acc_payment, acc_reward = 0, 0
        info = {}
        window = 5
        margin_ROI = 10 * np.ones(window)
        margin_cost = np.ones(window)
        margin_argmax = margin_ROI.argmax()

        for i in range(t_start, t_end):
            if 'USCB' in agent.name:
                action = agent.step(mode='eval')
                bid = min(action * self.values[i], budget - acc_payment)
            else:
                bid = min(agent.step(self.values[i]), budget - acc_payment)
            win = float(bid >= self.market_prices[i])
            payment = self.market_prices[i] * win
            reward = win * self.values[i]
            agent.update(reward, payment)
            acc_reward += reward
            acc_payment += payment

            if t_starts is not None:
                if i == t_starts[stage + 1]:
                    stage += 1
                acc_rewards[stage] += reward
                acc_payments[stage] += payment

        if t_starts is not None:
            return acc_rewards, acc_payments

        info['budget_stage'] = budget_stage

        margin_ROI = margin_ROI.mean()
        margin_cost = margin_cost.mean()
        return [acc_reward, acc_payment, margin_ROI, margin_cost], info


    def get_greedy_solution(self, budget, t_start=0, t_end=None, ROI=0):
        acc_reward, acc_payment = 0, 0
        t_end = self.epi_len if t_end is None else t_end

        values, market_prices = deepcopy(self.values[t_start:t_end]), deepcopy(self.market_prices[t_start:t_end])
        if market_prices.sum() < budget:
            print('Budget surplus')
        cost = market_prices / values
        idx = np.argsort(cost)
        market_prices = market_prices[idx]
        values = values[idx]

        margin_ROI, margin_cost = 10, 1
        info = {}
        j = -1

        for i in range(t_end - t_start):
            if values[i] < market_prices[i] * ROI or acc_payment + market_prices[i] > budget:
                j = i - 1
                break
            acc_payment += market_prices[i]

        if j >= 0:
            w = 1
            l = max(j-w+1, 0)
            margin_ROI = (values[l:j+1] / market_prices[l:j+1]).mean()
            margin_cost = market_prices[l:j+1].mean()
            acc_reward = values[:j].sum()

        return [acc_reward, acc_payment, margin_ROI, margin_cost], info

    def step(self, bid):
        next_obs = {
            "value": None,
            "payment": None
        }
        win = bid >= self.market_prices[self.t]
        reward = float(win * self.values[self.t])
        next_obs['payment'] = win * self.market_prices[self.t]
        info = {'market_price': self.market_prices[self.t]}

        self.t += 1
        done = False
        if self.t <= self.epi_len - 1:
            next_obs["value"] = self.values[self.t]
        else:
            done = True
        return next_obs, reward, done, info
