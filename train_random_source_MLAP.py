"""train MLAP agent with some sets of random source polices for OpenAI gym environments
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""

import argparse
import os
from pprint import pprint
import numpy as np
import yaml
import subprocess
from stable_baselines.results_plotter import ts2xy, load_results


_THRESH= {
    'RoboschoolHopper-v1': 2000,
    'LunarLanderContinuous-v2': 200,
    'Acrobot-v1': -100}


parser = argparse.ArgumentParser()

parser.add_argument('--num-jobs', help='number of parallel jobs', type=str, default=40)
parser.add_argument('--sources-dir', help='Directory where source policies are stored', type=str, required=True)
parser.add_argument('--algo', help='RL Algorithm', type=str, required=True, choices=['mlap-ppo2', 'mlap-sac'])
parser.add_argument('--num-set', help='Number of sets of source policies to sample randomly', type=int, default=5)
parser.add_argument('--num-sources', help='Number of source policies used in MLAP', type=int, default=4)
parser.add_argument('--SDW', help='Make master model state dependant', action='store_true', default=False)
parser.add_argument('--no-bias', help='Do not learn an auxiliary source policy', action='store_true', default=False)
parser.add_argument('--seed', help='Random generator seed', type=int, default=55)
parser.add_argument('--params-ranges', type=str, nargs='+', default=[], help='ranges of the samples of env dynamics',
                    required=True)

parser.add_argument('--env', type=str, help='environment ID',
                    choices=['RoboschoolHopper-v1', 'LunarLanderContinuous-v2', 'Acrobot-v1'], required=True)
parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')

args = parser.parse_args()
env_id = args.env
algo = args.algo
sources_dir = args.sources_dir
num_sources = args.num_sources
np.random.seed(args.seed)


def _get_random_source_policies():

    source_policies = []
    for source_path in os.listdir(sources_dir):
        if env_id in source_path and source_path[-1] == '1':
            path = sources_dir + source_path
            _, y = ts2xy(load_results(path), 'episodes')
            if np.mean(y[-100:]) > _THRESH[env_id]:
                source_policies.append('{}/{}.pkl'.format(path, env_id))

    if len(source_policies) < num_sources:
        raise ValueError('{} number of valid source policies is less than the requested number of sources {}'.format(
            source_policies, num_sources))

    source_policies = np.random.choice(source_policies, num_sources, replace=False)

    return source_policies.tolist()


with open('hyperparams/{}.yml'.format(algo), 'r') as f:
    all_hyperparams = yaml.load(f)
    hyperparams = all_hyperparams[env_id]


policy_kwargs = hyperparams['policy_kwargs']
policy_kwargs['SDW'] = args.SDW
policy_kwargs['no_bias'] = args.no_bias

exp_prefix = '{}sources-{}sets-'.format(num_sources, args.num_set)
exp_prefix += 'SDW' if args.SDW else 'SIW'
if args.no_bias:
    exp_prefix = 'no-bias'

for _ in range(args.num_set):

    policy_kwargs['source_policy_paths'] = _get_random_source_policies()

    # save the modified hyperparams
    with open('hyperparams/{}.yml'.format(algo), 'wb') as f:
        yaml.dump(all_hyperparams, f, default_flow_style=False,
                  explicit_start=True, allow_unicode=True, encoding='utf-8')

    pprint(hyperparams)

    train_envs_cmd = ['python', 'random_env_dynamic_train_cmd_gen.py',
                     '--num-samples', '100',
                     '--algo', algo,
                     '--seed', '0',
                     '--env', env_id,
                     '--log-folder', args.log_folder,
                     '--exp-prefix', exp_prefix,
                     '--params-ranges']

    for param in args.params_ranges:
        train_envs_cmd.append(param)

    subprocess.run(train_envs_cmd)

    subprocess.run(['parallel', '-a', '/tmp/out.txt', '--eta', '-j', args.num_jobs, '--load', '90%'])

