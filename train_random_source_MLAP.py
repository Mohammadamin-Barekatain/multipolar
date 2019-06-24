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

parser = argparse.ArgumentParser()

parser.add_argument('--num-jobs', help='number of parallel jobs', type=str, default=40)
parser.add_argument('--sources-dir', help='Directory where source policies are stored', type=str, required=True)
parser.add_argument('--algo', help='RL Algorithm', type=str, required=True, choices=['mlap-ppo2', 'mlap-sac'])
parser.add_argument('--num-set', help='Number of sets of source policies to sample randomly', type=int, default=5)
parser.add_argument('--num-sources', help='Number of source policies used in MLAP', type=int, default=4)
parser.add_argument('--SDW', help='Make master model state dependant', action='store_true', default=False)
parser.add_argument('--no-bias', help='Do not learn an auxiliary source policy', action='store_true', default=False)
parser.add_argument('--seed', help='Random generator seed', type=int, default=55)

parser.add_argument('--env', type=str, help='environment ID', choices=['RoboschoolHopper-v1', 'LunarLanderContinuous-v2'])
parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')

args = parser.parse_args()
env_id = args.env
algo = args.algo
np.random.seed(args.seed)


def hopper_get_random_source_policies():
    env_params = [(leg, foot, size, damp)
                   for leg, foot in [("041", "03"), ("044","033"), ("047","036"), ("05","039") , ("053","042")]
                   for size in  ["075", "08", "085", "09", "095", "1"]
                   for damp in ["05", "1", "15", "2", "25"]]

    sample_idxs = np.random.choice(list(range(len(env_params))), args.num_sources, replace=False)

    source_policies = []
    for i in sample_idxs:
        leg, foot, size, damp = env_params[i]

        exp_name = env_id + '_leg' + leg + '-foot' + foot + '-size' + size + '-damp' + damp + '_1'
        source_path = os.path.join(args.sources_dir, exp_name, env_id +'.pkl')

        source_policies.append(source_path)

    return source_policies


def lunar_get_random_source_policies():
    raise NotImplementedError


with open('hyperparams/{}.yml'.format(algo), 'r') as f:
    all_hyperparams = yaml.load(f)
    hyperparams = all_hyperparams[env_id]


policy_kwargs = hyperparams['policy_kwargs']
policy_kwargs['SDW'] = args.SDW
policy_kwargs['no_bias'] = args.no_bias

prefix_exp_name = 'SDW' if args.SDW else 'SIW'
if args.no_bias:
    prefix_exp_name = 'no-bias'

for _ in range(args.num_set):

    if env_id == 'RoboschoolHopper-v1':
        policy_kwargs['source_policy_paths'] = hopper_get_random_source_policies()
        bash_script = './hopperTrainEnvs.sh'

    elif env_id == 'LunarLanderContinuous-v2':
        policy_kwargs['source_policy_paths'] = lunar_get_random_source_policies()
        bash_script = './lunarTrainEnvs.sh'

    else:
        raise NotImplementedError

    # save the modified hyperparams
    with open('hyperparams/{}.yml'.format(algo), 'wb') as f:
        yaml.dump(all_hyperparams, f, default_flow_style=False,
                  explicit_start=True, allow_unicode=True, encoding='utf-8')

    pprint(hyperparams)

    train_envs_sh = [bash_script, algo, prefix_exp_name, args.log_folder]

    result = subprocess.run(train_envs_sh, stdout=subprocess.PIPE)

    subprocess.run(['parallel', '--eta', '-j', args.num_jobs, '--load', '80%', '--noswap'],
                   input=result.stdout)

