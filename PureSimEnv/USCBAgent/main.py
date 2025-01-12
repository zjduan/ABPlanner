import os
import numpy as np
import logging
import torch
from tqdm import tqdm
import datetime
from time import time
from trainer import run

if __name__ == '__main__':
    from utils import get_base_parser, str2bool
    parser = get_base_parser()
    parser.add_argument('--auction_episode_len', type=int, default=6000)
    parser.add_argument('--n_epoch', type=int, default=int(1000))
    parser.add_argument('--n_epi_per_epoch', type=int, default=1)
    parser.add_argument('--n_epi_in_test', type=int, default=1000)
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--agent_adjust_freq', type=int, default=1)

    args = parser.parse_args()
    run(args)
