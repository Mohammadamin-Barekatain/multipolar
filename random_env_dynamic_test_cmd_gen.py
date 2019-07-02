"""train MLAP agent with some sets of random source polices for OpenAI gym environments
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""

import argparse
import os
import numpy as np
from utils.wrappers import parse_params_ranges


parser = argparse.ArgumentParser()
parser.add_argument('--num-samples', help='number of env dynamics samples', type=int, required=True)
parser.add_argument('--test-dir', help='Directory where saved agents are stored', type=str, required=True)
parser.add_argument('--n-envs', help='number of parallel test processes', default=16, type=int)

parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--params-ranges', type=str, nargs='+', default=[], help='ranges of the samples of env dynamics',
                    required=True)
parser.add_argument('--exp-prefix',  help='(optional) prefix to experiment name, DO NOT USE _', type=str, default='')

args = parser.parse_args()
test_dir = args.test_dir
np.random.seed(args.seed)

assert (not ('_' in args.exp_prefix)), 'experiment prefix should not include _'


def truncate(n, decimals=0):
    # source: https://realpython.com/python-rounding/
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


param_sampler = parse_params_ranges(args.params_ranges)
with open('/tmp/out_test.txt', 'w') as f:
    for _ in range(args.num_samples):
        params = {param: truncate(sample_fn(), 2) for param, sample_fn in param_sampler.items()}

        for path in os.listdir(test_dir):
            if path[-1] != '1':
                continue

            for agent in ['{}.pkl'.format(path.split('_')[0]), 'best_model.pkl']:
                trained_agent = '{}/{}'.format(test_dir + path, agent)
                cmd = 'python test.py --trained-agent {} --n-envs {}'.format(trained_agent, args.n_envs)

                exp_name = '{}-{}'.format(args.exp_prefix, agent.split('_')[0])
                for param, val in params.items():
                    cmd += ' --{} {}'.format(param, val)
                    exp_name += '-' + '{}'.format(param.split('_')[0])

                    # remove . from the exp name
                    val = str(val)
                    exp_name += ''.join(val.split('.'))

                if exp_name[0] == '-':
                    exp_name = exp_name[1:]

                cmd += ' --exp-name ' + exp_name

                f.write(cmd + '\n')

