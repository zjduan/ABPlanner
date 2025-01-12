import os
import numpy as np
import random
import logging
import torch
import json
import shutil
import argparse
from scipy import signal

EPS = 1e-6

def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_tag', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--restart', action='store_true')
    return parser

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.set_save_dir(args)
        self.set_logger(args)
        self.set_seed(args)

    def set_save_dir(self, args):
        if args.save_dir:
            if not args.test and os.path.exists(args.save_dir):
                if args.restart:
                    shutil.rmtree(args.save_dir)
                else:
                    print('save path exist, continue training? [y/n]')
                    s = input()
                    if s == 'n':
                        shutil.rmtree(args.save_dir)
            elif args.test and not os.path.exists(args.save_dir):
                print(f'no checkpoint at {args.save_dir}!')
                exit()
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

    def set_logger(self, args):
        '''
        Write logs to checkpoint and console
        '''
        log_file = None
        filemode = 'a'
        if args.save_dir:
            log_file = 'test' if args.test else 'train'
            if args.test:
                filemode = 'w'
            if args.log_tag:
                log_file += f'_{args.log_tag}'
            log_file = os.path.join(args.save_dir, log_file + '.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode=filemode
        )
        if log_file is not None:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)
            logging.info(f'save log on {log_file}')

    def set_seed(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

    def save_model(self, args, save_variable_dict, tag=None):
        if args.save_dir:
            logging.info(f'save to {args.save_dir}')
            argparse_dict = vars(args)
            with open(os.path.join(args.save_dir, 'config.json'), 'w') as fjson:
                json.dump(argparse_dict, fjson)
            f_name = 'checkpoint' if tag is None else f'checkpoint-{tag}'
            torch.save(save_variable_dict, os.path.join(args.save_dir, f_name))

def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def projection_simplex_sort(v, z=1):
    assert z >= -EPS
    if abs(z) <= EPS:
        return np.zeros_like(v)
    n_features = v.shape[0]
    if n_features == 0:
        return 0
    elif n_features == 1:
        return np.ones(1) * z
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise ValueError('Unsupported value encountered.')
