"""train MLAP agent with some sets of random source polices for OpenAI gym environments
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""

import argparse
import os
import numpy as np
from utils.wrappers import parse_params_ranges


parser = argparse.ArgumentParser()
parser.add_argument('--num-test-episodes', help='Number of test episodes', type=int, required=True)
parser.add_argument('--test-dir', help='Directory where saved agents are stored', type=str, required=True)
parser.add_argument('--n-envs', help='Number of parallel test processes', default=1, type=int)
parser.add_argument('--action-noises', type=float, nargs='+',
                    help='List of stds of Gaussian noise injected to agent actions', required=True)
parser.add_argument('--observation-noises', type=float, nargs='+',
                    help='List of stds of Gaussian noise injected to agent observations', required=True)

parser.add_argument('--seed', help='Test random generator seed', type=int, default=0)
parser.add_argument('--exp-prefix',  help='(optional) prefix to experiment name, DO NOT USE _', type=str,
                    default='default')

args = parser.parse_args()
test_dir = args.test_dir
exp_prefix = args.exp_prefix

assert (not ('_' in exp_prefix)), 'experiment prefix should not include _'


with open('/tmp/out_test.txt', 'w') as f:

    for path in os.listdir(test_dir):
        trained_agent = '{}/{}.pkl'.format(test_dir + path, path.split('_')[0])

        cmd_base = 'python test.py --trained-agent {} --n-envs {} --seed {} --num-test-episodes {} --exp-name {} '.\
            format(trained_agent, args.n_envs, args.seed, args.num_test_episodes, exp_prefix)

        for ac_noise in args.action_noises:
            cmd = cmd_base + '--action-noise {}'.format(ac_noise)
            f.write(cmd + '\n')

        for ob_noise in args.observation_noises:
            cmd = cmd_base + '--observation-noise {}'.format(ob_noise)
            f.write(cmd + '\n')
