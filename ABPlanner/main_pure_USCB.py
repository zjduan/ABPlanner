import os
import numpy as np
import logging
import torch
from tqdm import tqdm
import datetime
from time import time
from trainer import run, Trainer

if __name__ == '__main__':
    from utils import get_base_parser, str2bool
    parser = get_base_parser()
    parser.add_argument('--env_name', type=str, choices=['SemiSimEnv', 'PureSimEnv'], default='PureSimEnv')
    parser.add_argument('--fix_value', type=str2bool, default=False)
    parser.add_argument('--init_budget_plan', type=str, choices=['random', 'average', 'RLPlanner', 'PID', 'USCB'], default='USCB')
    parser.add_argument('--n_obs_row', type=int, choices=[3], default=3)
    parser.add_argument('--low_agent', type=str, choices=['PID', 'Greedy', 'USCB', 'LP'], default='USCB')
    parser.add_argument('--n_epoch', type=int, default=int(100))
    parser.add_argument('--n_task_per_epoch', type=int, default=10)
    parser.add_argument('--n_task_in_test', type=int, default=1000)
    parser.add_argument('--n_epi_per_task', type=int, default=8)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--reward_fn', type=str, choices=['dif', 'sum'], default='dif')
    parser.add_argument('--action_dim', type=int, default=-1)
    parser.add_argument('--verbose_test', type=str2bool, default=False)

    parser.add_argument('--auction_episode_len', type=int, default=6000)
    parser.add_argument('--n_stage', type=int, default=6)
    parser.add_argument('--fix_budget', type=str2bool, default=True)
    parser.add_argument('--test_data_dir', type=str, default=None)

    # For USCB Agent
    parser.add_argument('--agent_save_dir', type=str, default='PureSimEnv/USCBAgent/result/')
    parser.add_argument('--agent_adjust_freq', type=int, default=1)

    args = parser.parse_args()
    run(args)
