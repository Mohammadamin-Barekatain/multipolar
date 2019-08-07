"""train MLAP agent with some sets of random source polices for OpenAI gym environments
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""

import argparse
import numpy as np
from utils.wrappers import parse_params_ranges

parser = argparse.ArgumentParser()
parser.add_argument('--num-samples', help='number of env dynamics samples', type=int, required=True)
parser.add_argument('--algo', help='RL Algorithm', type=str, required=True)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--env', type=str, help='environment ID',
                    choices=['RoboschoolHopper-v1', 'LunarLanderContinuous-v2', 'Acrobot-v1'], required=True)
parser.add_argument('--params-ranges', type=str, nargs='+', default=[], help='ranges of the samples of env dynamics',
                    required=True)
parser.add_argument('--exp-prefix',  help='(optional) prefix to experiment name, DO NOT USE _', type=str, default='')

parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')

args = parser.parse_args()
env = args.env
np.random.seed(args.seed)

assert (not ('_' in args.exp_prefix)), 'experiment prefix should not include _'


def truncate(n, decimals=0):
    # source: https://realpython.com/python-rounding/
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


param_sampler = parse_params_ranges(args.params_ranges)
with open('/tmp/out.txt', 'w') as f:
    for _ in range(args.num_samples):
        params = {param: truncate(sample_fn(), 2) for param, sample_fn in param_sampler.items()}

        for seed in [1000, 2000, 3000]:

            cmd = 'python train.py --env {} --algo {} --seed {} --log-folder {} --play 1000 --no-tensorboard'.format(
                env, args.algo, seed, args.log_folder)

            if seed != 1000:
                cmd += ' --no-tensorboard'

            exp_name = args.exp_prefix
            for param, val in params.items():
                cmd += ' --{} {}'.format(param, val)
                if env == 'LunarLanderContinuous-v2' or env == 'Acrobot-v1':
                    parts_param = [p.capitalize() for p in param.split('_') if not p.isdigit()]
                    exp_name += '-' + '{}'.format(''.join(parts_param))

                else:
                    exp_name += '-' + '{}'.format(param.split('_')[0])

                # remove . from the exp name
                val = str(val)
                exp_name += ''.join(val.split('.'))

            if exp_name[0] == '-':
                exp_name = exp_name[1:]

            cmd += ' --exp-name ' + exp_name

            f.write(cmd + '\n')
