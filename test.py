"""Test agents on OpenAI gym environments
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""

import argparse
import os
from pprint import pprint

import gym
import numpy as np
import pandas as pd
import yaml
from stable_baselines.common import set_global_seeds

from utils import parse_unknown_args, create_test_env
from utils.callbacks import VideoRecorder
from utils.policies import ALGOS


def evaluate_model(model, env, num_test_episodes, deterministic=True, action_noise=0, observation_noise=0):
    """ returns a np array of length num_test_episodes, specifying the accumulated reward an agent got on the given env.

    :param model: (obj) stable_baseline model instance
    :param env: (obj) a Gym Environment instance
    :param num_test_episodes: (int) number of test episodes
    :param deterministic: (bool) weather to sample actions deterministicly (True) or randomly (False) from the policy.
    :param action_noise: (float) std of Gaussian noise injected to agents actions
    :param observation_noise: (float) std of Gaussian noise injected to agents observations.
    """
    obs = env.reset()
    n_envs = len(obs)
    ep_rewards = []
    running_rewards = np.zeros(n_envs, dtype=np.float32)

    while num_test_episodes > len(ep_rewards):

        if observation_noise > 0:
            obs += np.random.normal(scale=observation_noise, size=(n_envs,)+env.observation_space.shape)

        action, _ = model.predict(obs, deterministic=deterministic)

        if action_noise > 0:
            action += np.random.normal(scale=action_noise, size=(n_envs,)+env.action_space.shape)

        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, done, _ = env.step(action)
        running_rewards += reward

        for i in range(n_envs):
            if done[i] and num_test_episodes > len(ep_rewards):
                ep_rewards.append(running_rewards[i])
                running_rewards[i] = 0

    return np.array(ep_rewards, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Any extra args will be used for modifying environment dynamics')
    parser.add_argument('--trained-agent', help='Path to a pretrained agent', default=None, type=str, required=True)
    parser.add_argument('--exp-name', help='experiment name, DO NOT USE _', default=None, type=str, required=True)
    parser.add_argument('--n-envs', help='number of parallel test processes', default=16, type=int)
    parser.add_argument('-nte', '--num-test-episodes', help='number of test episodes', default=100, type=int)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--action-noise', help='std of Gaussian noise injected to agent actions', type=float, default=0)
    parser.add_argument('--observation-noise', help='std of Gaussian noise injected to agents observations',
                        type=float, default=0)

    parser.add_argument('--play', help='Length of the video of the agent performing actions on the env (-1 = disabled)',
                        default=1000, type=int)
    parser.add_argument('--sample-actions-randomly', action='store_true', default=False,
                        help='Sample actions randomly from the policy')

    args, env_params = parser.parse_known_args()
    env_params = parse_unknown_args(env_params)

    seed = args.seed
    trained_agent_path = args.trained_agent
    deterministic = not args.sample_actions_randomly

    assert trained_agent_path.endswith('.pkl') and os.path.isfile(trained_agent_path), \
        "The trained_agent must be a valid path to a .pkl file"

    set_global_seeds(seed)
    algo = trained_agent_path.split('/')[1].split('_')[0]
    env_id = trained_agent_path.split('/')[2].split('_')[0]

    # set-up saving/logging paths
    parent_dir_idx = trained_agent_path.rfind('/')
    log_path = "{}/{}".format(trained_agent_path[:parent_dir_idx], 'results')

    exp_name = args.exp_name
    assert (not ('_' in exp_name)), 'experiment name should not include _'
    save_path = os.path.join(log_path, "{}_{}".format(exp_name, seed))
    os.makedirs(save_path, exist_ok=True)

    params_path = "{}/{}".format(trained_agent_path[:parent_dir_idx], env_id)
    assert os.path.isdir(params_path), 'invalid params path %s' % params_path

    # Load hyperparameters from yaml file
    with open('hyperparams/{}.yml'.format(algo), 'r') as f:
        if 'NoFrameskip' in env_id:
            hyperparams = yaml.load(f)['atari']
        else:
            hyperparams = yaml.load(f)[env_id]
        del hyperparams['n_timesteps']


    # Optional Frame-stacking
    n_stack = 1
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        del hyperparams['frame_stack']

    # normalizing env?
    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    # create the test env
    env_hyperparams = {'normalize': normalize, 'n_stack': n_stack, 'normalize_kwargs': normalize_kwargs}
    env = create_test_env(env_id, n_envs=args.n_envs, stats_path=params_path, seed=seed, hyperparams=env_hyperparams,
                          env_params=env_params, should_render=False)

    # load the pretrained agent
    print("Loading pretrained agent")
    model = ALGOS[algo].load(trained_agent_path, env=env, verbose=1)

    # test the pretrained agent
    print("=" * 10, "TESTING", env_id, "=" * 10)
    if len(env_params):
        print("environment parameters")
        pprint(env_params)

    ep_rewards = evaluate_model(model, env, num_test_episodes=args.num_test_episodes, deterministic=deterministic,
                                action_noise=args.action_noise, observation_noise=args.observation_noise)

    print("mean reward: ", ep_rewards.mean())
    print("std reward: ", ep_rewards.std())

    # save the results
    # Create DataFrame
    results_df = {'mean_reward': ep_rewards.mean(), 'std_reward': ep_rewards.std(), 'rewards': ep_rewards}
    results_df = pd.DataFrame(results_df)

    # dump as csv file:
    name_postfix = 'acNoise{}_obNoise{}'.format(args.action_noise, args.observation_noise)
    # remove . from the name_postfix
    name_postfix = ''.join(name_postfix.split('.'))

    save_file = '{}/results_{}.csv'.format(save_path, name_postfix)
    results_df.to_csv(save_file, sep=",", index=False)
    print("Saved results to {}".format(save_file))

    # save video
    if args.play > 0:
        name = "agent_{}_env_{}_{}".format(trained_agent_path.split('/')[2].split('_')[1], exp_name, name_postfix)
        record = VideoRecorder(env_id, save_path, env_hyperparams, params_path, args.play, interval=1, seed=seed,
                               env_params=env_params, deterministic=deterministic, name_prefix=name).callback
        record({'self': model}, None)

