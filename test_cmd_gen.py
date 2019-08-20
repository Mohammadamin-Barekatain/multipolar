"""train MLAP agent with some sets of random source polices for OpenAI gym environments
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""

import argparse
import os
from utils import extract_param_val
from utils.wrappers import parse_params_ranges

_DISCRETE_ENVS=['Acrobot-v1']

parser = argparse.ArgumentParser()
parser.add_argument('--num-test-episodes', help='Number of test episodes', type=int, required=True)
parser.add_argument('--test-dir', help='Directory where saved agents are stored', type=str, required=True)
parser.add_argument('--n-envs', help='Number of parallel test processes', default=1, type=int)
parser.add_argument('--action-noises', type=float, nargs='+',
                    help='List of stds of Gaussian noise injected to agent actions', required=True)
parser.add_argument('--observation-noises', type=float, nargs='+',
                    help='List of stds of Gaussian noise injected to agent observations', required=True)
parser.add_argument('--override', action='store_true', default=False, help='override the test results if they exist')

parser.add_argument('--seed', help='Test random generator seed', type=int, default=0)
parser.add_argument('--exp-prefix',  help='(optional) prefix to experiment name, DO NOT USE _', type=str,
                    default='default')

args = parser.parse_args()
test_dir = args.test_dir
exp_prefix = args.exp_prefix

assert (not ('_' in exp_prefix)), 'experiment prefix should not include _'


def exists(test_dir, ac_noise, ob_noise):
    name_postfix = 'acNoise{}_obNoise{}'.format(ac_noise, ob_noise)
    # remove . from the name_postfix
    name_postfix = ''.join(name_postfix.split('.'))

    file = os.path.join(test_dir, 'results/{}_{}/results_{}.csv'.format(exp_prefix, args.seed, name_postfix))

    return os.path.isfile(file)


with open('/tmp/out_test.txt', 'w') as f:

    for path in os.listdir(test_dir):
        trained_agent = '{}/{}.pkl'.format(test_dir + path, path.split('_')[0])

        cmd_base = 'python test.py --trained-agent {} --n-envs {} --seed {} --num-test-episodes {} --exp-name {} '.\
            format(trained_agent, args.n_envs, args.seed, args.num_test_episodes, exp_prefix)

        param_val = extract_param_val(path.split('_')[1])
        counter = {'LINK_MASS': 0, 'LINK_COM_POS': 0, 'LINK_LENGTH': 0}
        for param, val in param_val:
            if param in ['leg', 'foot', 'thigh', 'torso']:
                param += '_length'

            if sum(1 for c in param if c.isupper()) > 0:
                p = [param[0]]
                for c in param[1:]:
                    if c.isupper():
                        p.append('_')
                    p.append(c)
                param = ''.join(p).upper()
                if param in counter:
                    counter[param] += 1
                    param += '_{}'.format(counter[param])

            cmd_base += '--{} {} '.format(param, val)

        if path.split('_')[0] not in _DISCRETE_ENVS:
            for ac_noise in args.action_noises:
                cmd = cmd_base + '--action-noise {}'.format(ac_noise)
                if args.override or (not exists(test_dir + path, ac_noise, '0')):
                    f.write(cmd + '\n')

        for ob_noise in args.observation_noises:
            cmd = cmd_base + '--observation-noise {}'.format(ob_noise)
            if args.override or (not exists(test_dir + path, '0', ob_noise)):
                f.write(cmd + '\n')
