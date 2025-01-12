import os
import numpy as np
import logging
import torch
from tqdm import tqdm
import datetime
from time import time
from copy import deepcopy
from IPython import embed

from utils import BaseTrainer, EPS
from replay_buffer import ReplayBuffer
from PPO import PPOAgent
from PureSimEnv.pure_env import BudgetAllocationEnv as PureSimEnv
from PureSimEnv.PID_agent import PIDAgent
from PureSimEnv.USCBAgent.USCB_agent import USCBAgent as USCBAgent

class Trainer(BaseTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        low_agent = None
        if args.low_agent == 'PID':
            low_agent = PIDAgent()
        elif 'USCB' in args.low_agent:
            low_agent = USCBAgent(act_freq=args.agent_adjust_freq)
            if args.agent_save_dir:
                ckpt_path = os.path.join(args.agent_save_dir, 'checkpoint')
                if os.path.exists(ckpt_path):
                    logging.info(f'load agent checkpoint from {ckpt_path}')
                    ckpt = torch.load(ckpt_path)
                    low_agent.load_save_dict(ckpt, mode='eval')

        self.env = PureSimEnv(args.n_stage,
                                auction_episode_len=args.auction_episode_len,
                                fix_budget=args.fix_budget,
                                fix_value=args.fix_value,
                                agent=low_agent,
                                n_obs_row=args.n_obs_row,
                                data_dir=args.test_data_dir)

        self.agent = PPOAgent(args.n_stage,
                              [args.n_obs_row, args.n_stage],
                              args.action_dim,
                              args.n_epi_per_task,
                              args.device)
        self.replay_buffer = ReplayBuffer(args.n_task_per_epoch,
                                          args.n_epi_per_task,
                                          args.n_stage,
                                          self.agent.action_dim,
                                          gamma=self.agent.gamma)
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

    def _get_init_obs_matrix(self, args, env, budget):
        if args.init_budget_plan == 'USCB' or args.init_budget_plan == 'PID':
            assert args.env_name == 'PureSimEnv'
            next_obs_matrix, next_budget, info = env.step(0)

            budget_stage = deepcopy(next_obs_matrix[2])
            budget_stage[-1] += budget - next_obs_matrix[2].sum()
            next_obs_matrix[0] = deepcopy(budget_stage)

            obs_matrix = next_obs_matrix
            budget = next_budget
        else:
            assert args.init_budget_plan in ['average', 'random', 'LP']
            init_budget_stage = env.get_init_budget_stage(budget, args.init_budget_plan)
            next_obs_matrix, next_budget, info = env.step(init_budget_stage)

            obs_matrix = next_obs_matrix
            budget = next_budget

        return obs_matrix, budget


    def train(self, args):
        self.load(args)
        env = self.env
        env.train()
        n_task = args.n_task_per_epoch
        for epoch in tqdm(range(self.start_epoch, args.n_epoch)):
            obs_list, budget_list, action_list, reward_list, value_list, logp_list = \
                [], [], [], [], [], []
            all_returns = []
            for task in range(n_task):
                _, budget = env.reset()
                self.agent.reset()
                self.agent.train()

                obs_matrix, budget = self._get_init_obs_matrix(args, env, budget)
                return_stage = obs_matrix[1]
                all_returns.append(return_stage.sum())

                for episode in range(1, args.n_epi_per_task):
                    budget_stage, agent_info = self.agent.step(obs_matrix, budget, test=False)
                    action, value, logp = agent_info['action'], agent_info['value'], agent_info['logp']

                    next_obs_matrix, next_budget, info = env.step(budget_stage)
                    return_stage = next_obs_matrix[1]
                    all_returns.append(return_stage.sum())

                    if args.reward_fn == 'sum':
                        reward = all_returns[-1]
                    else:
                        reward = all_returns[-1] - all_returns[-2]

                    obs_list.append(obs_matrix)
                    budget_list.append(budget)
                    action_list.append(action)
                    reward_list.append(reward)
                    value_list.append(value)
                    logp_list.append(logp)

                    obs_matrix = next_obs_matrix
                    budget = next_budget

            self.replay_buffer.fit(obs_list, budget_list, action_list,
                                   reward_list, value_list, logp_list)
            info = self.agent.learn(self.replay_buffer.get())

            if epoch % args.log_freq == 0:
                np.set_printoptions(suppress=True)
                print('\nobs_list\n', np.around(np.array(obs_list[-(args.n_epi_per_task-1):]), 4),
                      '\nbudget_stage\n', np.around(budget_stage, 4),
                      '\naction\n', np.around(action_list[-(args.n_epi_per_task-1):], 4),
                      '\nreward\n', np.around(np.array(reward_list[-(args.n_epi_per_task-1):]), 4),
                      '\nall_returns\n', np.around(np.array(all_returns[-(args.n_epi_per_task):]), 4),
                      '\nstd\n', self.agent.actor_critic.std_scaler * torch.exp(self.agent.actor_critic.log_std))

                real_all_returns_mean = (np.array(all_returns) / env.scaler).reshape(n_task, -1).mean(0)
                logging.info(f"Train: epoch={epoch}, real_all_returns_mean={list(real_all_returns_mean)},\n")
                self.save(args, epoch)

        self.test(args, load=False)

    def test(self, args, load=False):
        verbose = args.verbose_test
        if load is True:
            self.load(args)
        env = self.env
        env.test()

        n_task = args.n_task_in_test
        if n_task is None or n_task <= 0:
            n_task = env.n_task
        obs_list, budget_list = [], []
        all_returns, average_returns, random_returns, oracle_returns = [], [], [], []
        all_budget_stages = []
        initial_returns = []
        for task in tqdm(range(n_task)):
            _, budget = env.reset()
            self.agent.reset()
            self.agent.eval()

            obs_matrix, budget = self._get_init_obs_matrix(args, env, budget)
            init_budget_stage = obs_matrix[0]
            return_stage = obs_matrix[1]
            all_returns.append(return_stage.sum()/env.scaler)
            initial_returns.append(return_stage.sum()/env.scaler)
            all_budget_stages.append(init_budget_stage)

            for episode in range(1, args.n_epi_per_task):
                budget_stage, agent_info = self.agent.step(obs_matrix, budget, test=True)

                # if verbose:
                #     average_returns.append(env.get_average_result(budget)/env.scaler)
                #     random_returns.append(env.get_random_result(budget)/env.scaler)
                    # oracle_returns.append(env.get_greedy_result(budget)/env.scaler)

                tmp = env.get_stage_result(init_budget_stage)
                initial_returns.append(tmp[0].sum()/env.scaler)

                next_obs_matrix, next_budget, info = env.step(budget_stage)
                return_stage = next_obs_matrix[1]

                all_returns.append(return_stage.sum()/env.scaler)
                all_budget_stages.append(budget_stage)

                if episode > 0:
                    obs_list.append(obs_matrix)
                    budget_list.append(budget)

                obs_matrix = next_obs_matrix
                budget = next_budget

            if task % 100 == 0 or task + 1 == n_task:
                np.set_printoptions(suppress=True)
                print('\nobs_list\n', np.around(np.array(obs_list[-(args.n_epi_per_task-1):]), 4),
                      '\nbudget_stage\n', np.around(budget_stage, 4),
                      '\nall_returns\n', np.around(np.array(all_returns[-(args.n_epi_per_task-1):]), 4))

                all_returns_mean = np.array(all_returns).reshape(task+1, -1).mean(0)
                initial_returns_mean = np.array(initial_returns).reshape(task+1, -1).mean(0)

                budget_stages_mean = np.array(all_budget_stages).reshape(task+1, args.n_epi_per_task, -1).mean(0)

                logging.info(f"Test:\n"
                             f"opalgo_returns={list(all_returns_mean)}, \n"
                             f"initia_returns={list(initial_returns_mean)}, \n"
                             )
                # if verbose:
                #     average_returns_mean = np.array(average_returns).reshape(task+1, -1).mean(0)
                #     random_returns_mean = np.array(random_returns).reshape(task+1, -1).mean(0)
                #     logging.info(
                #         f"averag_returns={list(average_returns_mean)}, \n"
                #         f"random_returns={list(random_returns_mean)}, \n"
                #         f"budget_stages={list(budget_stages_mean)}\n"
                #     )

        if verbose and args.save_dir is not None:
            np.save(os.path.join(args.save_dir, 'budget_stages.npy'), budget_stages_mean)

def run(args):
    t0 = time()

    trainer = Trainer(args)
    if args.test:
        trainer.test(args, load=True)
    else:
        trainer.train(args)

    time_used = time() - t0
    logging.info(f'Time Cost={datetime.timedelta(seconds=time_used)}')
