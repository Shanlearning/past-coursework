# -*- coding: UTF-8 -*-
'''
import os
os.chdir("C:\\Users\\zhong\\Dropbox\\statistics\\CS 790\\assignment3\\starter_code_torch\\code")
'''

import argparse
import numpy as np
import torch
import gym

from policy_gradient import PolicyGradient
from config import get_config
import random
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', required=True, type=str,
                    choices=['cartpole', 'pendulum', 'cheetah'])
parser.add_argument('--baseline', dest='use_baseline', action='store_true')
parser.add_argument('--no-baseline', dest='use_baseline', action='store_false')
parser.add_argument('--seed', type=int, default=1)
parser.set_defaults(use_baseline=True)

if __name__ == '__main__':
    
    seed= 3
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
        
    config = get_config('cartpole', False , seed)

    env = gym.make("CartPole-v1")
    # train model
    model = PolicyGradient(env, config, seed)
    model.run()
