import os
import sys
import numpy as np
import logging
import torch
from tqdm import tqdm
import datetime
from time import time
from IPython import embed

from USCB_agent import USCBAgent
from utils import BaseTrainer, EPS
from PureSimEnv.spa_env import SPAEnv


class Trainer(BaseTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.env = SPAEnv(args.auction_episode_len)

        self.agent = USCBAgent(dim_action=1,
                               act_freq=args.agent_adjust_freq,
                               device=args.device)
        self.start_epoch = 0

    def save(self, args, epoch, tag=None):
        if not args.save_dir:
            return
        save_variable_dict = {
            **self.agent.get_save_dict(),
            'start_epoch': epoch+1,
        }
        self.save_model(args, save_variable_dict, tag=tag)

    def load(self, args, tag=None):
        fname = 'checkpoint' if tag is None else f'checkpoint-{tag}'
        try:
            ckpt_path = os.path.join(args.save_dir, fname)
        except:
            return
        if os.path.exists(ckpt_path):
            logging.info(f'load checkpoint from {ckpt_path}')
            ckpt = torch.load(ckpt_path)
            self.agent.load_save_dict(ckpt)
            self.start_epoch = ckpt['start_epoch']

    def train(self, args):
        self.load(args)
        env = self.env
        agent = self.agent

        acc_rewards = []
        speeds = []
        remain_budgets = []
        for epoch in tqdm(range(self.start_epoch, args.n_epoch)):
            for epi in range(args.n_epi_per_epoch):
                acc_reward = 0
                speed = 0

                epi_start = np.random.randint(0, env.epi_len - agent.act_freq)
                epi_len = np.random.randint(agent.act_freq, env.epi_len)
                init_obs = env.reset(sample_budget=False, epi_start=epi_start)
                init_obs["budget"] = np.random.uniform(0, 200)

                remain_budget = init_obs["budget"]
                agent.reset(epi_len, remain_budget, cur_T=epi_start)
                agent.train()

                obs= init_obs

                step = 0
                while True:
                    pv_value = obs["value"]
                    action = agent.step(mode="train")
                    bid = min(action * pv_value, remain_budget)
                    if bid < remain_budget:
                        speed += 1

                    next_obs, reward, done, info = env.step(bid)
                    if remain_budget < EPS:
                        done = True

                    payment = next_obs["payment"]
                    remain_budget -= payment
                    acc_reward += reward

                    step += 1
                    if step + 1 == epi_len:
                        done = True

                    agent.update(reward, payment, done, mode='train')

                    obs = next_obs

                    if done == True:
                        break

                acc_rewards.append(acc_reward)
                speeds.append(speed)
                remain_budgets.append(remain_budget)

            if epoch % args.log_freq == 0 or epoch + 1 == args.n_epoch:
                acc_rewards = np.array(acc_rewards)
                logging.info(f"Train: epoch={epoch}, acc_reward={acc_rewards.mean()}")
                self.save(args, epoch)

                speeds = np.array(speeds)
                remain_budgets = np.array(remain_budgets)
                logging.info(f"{agent.actor.log_std.item()}, {speeds.mean()}, {remain_budgets.mean()}")
                acc_rewards = []
                speeds = []
                remain_budgets = []

        self.test(args, load=False)

    def test(self, args, load=False):
        if load is True:
            self.load(args)

        env = self.env
        agent = self.agent
        acc_rewards = []
        speeds = []
        remain_budgets = []
        n_epi = args.n_epi_in_test
        logging.info('Begin to test')
        for epi in tqdm(range(n_epi)):
            acc_reward = 0
            speed = 0

            init_obs = env.reset()
            remain_budget = init_obs["budget"]
            epi_len = env.epi_len

            agent.reset(epi_len, remain_budget)
            agent.eval()

            obs = init_obs
            while True:
                pv_value = obs["value"]
                action = agent.step(mode="eval")
                bid = min(action * pv_value, remain_budget)
                if bid < remain_budget:
                    speed += 1

                next_obs, reward, done, info = env.step(bid)
                if remain_budget < EPS:
                    done = True

                payment = next_obs["payment"]
                remain_budget -= payment
                acc_reward += reward

                agent.update(reward, payment, done, mode='eval')

                obs = next_obs
                if done == True:
                    break

            acc_rewards.append(acc_reward)
            speeds.append(speed)
            remain_budgets.append(remain_budget)

        acc_rewards = np.array(acc_rewards)
        speeds = np.array(speeds)
        remain_budgets = np.array(remain_budgets)
        logging.info(f"Test: acc_reward={acc_rewards.mean()}, speed={speeds.mean()}, {remain_budgets.mean()}")


def run(args):
    t0 = time()

    trainer = Trainer(args)
    if args.test:
        trainer.test(args, load=True)
    else:
        trainer.train(args)

    time_used = time() - t0
    logging.info(f'Time Cost={datetime.timedelta(seconds=time_used)}')
