"""Test agents on OpenAI gym environments
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""
import os
import gym
import yaml
import argparse
import numpy as np
import pandas as pd

from pprint import pprint
from stable_baselines.common import set_global_seeds
from utils import ALGOS, parse_unknown_args, create_test_env
from utils.callbacks import VideoRecorder

def evaluate_model(model, env, num_test_episodes=100, deterministic=True):
    """ returns a np array of length num_test_episodes, specifying the accumulated reward an agent got on the given env.

    :param model: (obj) stable_baseline model instance
    :param env: (obj) a Gym Environment instance
    :param num_test_episodes: (int) number of test episodes
    :param deterministic: (bool) weather to sample actions deterministicly (True) or randomly (False) from the policy.
    """
    obs = env.reset()
    n_envs = len(obs)
    ep_rewards = []
    running_rewards = np.zeros(n_envs, dtype=np.float32)

    while num_test_episodes > len(ep_rewards):

        action, _ = model.predict(obs, deterministic=deterministic)
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, done, _ = env.step(action)
        running_rewards += reward

        for i in range(n_envs):
            if done[i] and num_test_episodes > len(ep_rewards):
                #ToDo: Maybe for env using VecNormalize, the mean reward should be normalized reward
                ep_rewards.append(running_rewards[i])
                running_rewards[i] = 0

    return np.array(ep_rewards, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Any extra args will be used for modifying environment dynamics')
    parser.add_argument('--trained-agent', help='Path to a pretrained agent', default=None, type=str, required=True)

    parser.add_argument('--exp-name', help='experiment name, DO NOT USE _', default=None, type=str, required=True)
    parser.add_argument('--n-envs', help='number of processes', default=32, type=int)
    parser.add_argument('-nte', '--num-test-episodes', help='number of test episodes', default=100, type=int)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)

    parser.add_argument('--play', help='Length of the video of the agent performing actions on the env (-1 = disabled)',
                        default=1000, type=int)
    parser.add_argument('--sample-actions-randomly', action='store_true', default=False,
                        help='Sample actions randomly from the policy')

    args, env_params = parser.parse_known_args()
    env_params = parse_unknown_args(env_params)

    seed = args.seed
    exp_name = args.exp_name
    trained_agent_path = args.trained_agent
    deterministic = not args.sample_actions_randomly

    assert trained_agent_path.endswith('.pkl') and os.path.isfile(trained_agent_path), \
        "The trained_agent must be a valid path to a .pkl file"

    set_global_seeds(seed)
    algo = trained_agent_path.split('/')[1]
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
    # Policy should not be changed
    del hyperparams['policy']
    model = ALGOS[algo].load(trained_agent_path, env=env, verbose=1, **hyperparams)

    # test the pretrained agent
    print("=" * 10, "TESTING", env_id, "=" * 10)
    if len(env_params):
        print("environment parameters")
        pprint(env_params)

    ep_rewards = evaluate_model(model, env, args.num_test_episodes, deterministic=deterministic)

    print("mean reward: ", ep_rewards.mean())
    print("std reward: ", ep_rewards.std())

    # save the results
    # Create DataFrame
    results_df = {'mean_reward': ep_rewards.mean(), 'std_reward': ep_rewards.std(), 'rewards': ep_rewards}
    results_df = pd.DataFrame(results_df)

    # dump as csv file:
    save_file = '{}/results.csv'.format(save_path)
    results_df.to_csv(save_file, sep=",", index=False)
    print("Saved results to {}".format(save_file))

    # save video
    if args.play > 0:
        name = "agent_{}_env_{}".format(trained_agent_path.split('/')[2].split('_')[1], exp_name)
        record = VideoRecorder(env_id, save_path, env_hyperparams, params_path, args.play, interval=1, seed=seed,
                               env_params=env_params, deterministic=deterministic, name_prefix=name).callback
        record({'self': model}, None)









