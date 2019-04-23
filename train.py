"""train an agent for OpenAI gym environments
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX

Small parts of this script has been copied from https://github.com/araffin/rl-baselines-zoo
"""

import argparse
import difflib
import os
from collections import OrderedDict
from pprint import pprint

import gym
import numpy as np
import yaml
from stable_baselines import logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.bench import Monitor
from utils.wrappers import ModifyEnvParams
from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, load_group_results, parse_unknown_args, create_test_env
from utils.plot import plot_results

parser = argparse.ArgumentParser(description='Any extra args will be used for modifying environment dynamics')
parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID')
parser.add_argument('--algo', help='RL Algorithm', default='ppo2', type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--exp-name',  help='(optional) experiment name, DO NOT USE _', type=str, default=None)
parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1, type=int)

parser.add_argument('--trained-agent', help='Path to a pretrained agent to continue training', default='', type=str)
parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
parser.add_argument('--play', help='Length of gif of the final trained agent (-1 = disabled)', default=-1, type=int)

parser.add_argument('log-outputs', help='Save the putputs instead of diplying them', default=False)
parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
parser.add_argument('--no-monitor', help='do not monitor training', action='store_true', default=False)
parser.add_argument('--no-tensorboard', help='do not create tensorboard', action='store_true', default=False)
parser.add_argument('--no-plot', help='do not plot the results', action='store_true', default=False)
# ToDo: get path for loading a spesific hyperparams.
# ToDo: add saving to wandb
# ToDo: support changing environments for Atari
args, env_params = parser.parse_known_args()
env_params = parse_unknown_args(env_params)

if args.trained_agent != "":
    assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
        "The trained_agent must be a valid path to a .pkl file"

set_global_seeds(args.seed)
env_id = args.env
registered_envs = set(gym.envs.registry.env_specs.keys())
# If the environment is not found, suggest the closest match
if env_id not in registered_envs:
    closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
    raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

exp_name = args.exp_name
log_path = "{}/{}/".format(args.log_folder, args.algo)

if exp_name:
    assert (not ('_' in exp_name)), 'experiment name should not include _'
    save_path = os.path.join(log_path,
                             "{}_{}_{}".format(env_id, exp_name, get_latest_run_id(log_path, env_id, exp_name) + 1))
else:
    save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))

if args.log_outputs:
    # Log the outputs
    logger.configure(folder=save_path, format_strs=['log'])

params_path = "{}/{}".format(save_path, env_id)
os.makedirs(params_path, exist_ok=True)
tensorboard_log = None if args.no_tensorboard else save_path
monitor_log = None if args.no_monitor else save_path

is_atari = 'NoFrameskip' in env_id

print("=" * 10, env_id, "=" * 10)

# Load hyperparameters from yaml file
with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
    if is_atari:
        hyperparams = yaml.load(f)['atari']
    else:
        hyperparams = yaml.load(f)[env_id]

    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        hyperparams['n_timesteps'] = args.n_timesteps
    n_timesteps = int(hyperparams['n_timesteps'])

# Sort hyperparams that will be saved
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
pprint(saved_hyperparams)
if len(env_params):
    print("environment parameters")
    pprint(env_params)

n_envs = hyperparams.get('n_envs', 1)

print("Using {} environments".format(n_envs))

# Create learning rate schedules for ppo2 and sac
if args.algo in ["ppo2", "sac"]:
    for key in ['learning_rate', 'cliprange']:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split('_')
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], float):
            hyperparams[key] = constfn(hyperparams[key])
        else:
            raise ValueError('Invalid valid for {}: {}'.format(key, hyperparams[key]))

normalize = False
normalize_kwargs = {}
if 'normalize' in hyperparams.keys():
    normalize = hyperparams['normalize']
    if isinstance(normalize, str):
        normalize_kwargs = eval(normalize)
        normalize = True
    del hyperparams['normalize']

# Delete keys so the dict can be pass to the model constructor
if 'n_envs' in hyperparams.keys():
    del hyperparams['n_envs']
del hyperparams['n_timesteps']

# Create the environment and wrap it if necessary
if is_atari:
    print("Using Atari wrapper")
    env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)
    if not args.no_monitor:
        print("WARNING: monitor is not supported yet for atari env")
elif args.algo in ['dqn', 'ddpg']:
    if hyperparams.get('normalize', False):
        print("WARNING: normalization not supported yet for DDPG/DQN")
    env = gym.make(env_id)
    if len(env_params) > 0:
        env = ModifyEnvParams(env, **env_params)
    env.seed(args.seed)
    if not args.no_monitor:
        env = Monitor(env, monitor_log, allow_early_resets=True)
else:
    if n_envs == 1:
        env = DummyVecEnv([make_env(env_id, 0, args.seed, monitor_log, env_params)])
    else:
        env = SubprocVecEnv([make_env(env_id, i, args.seed, monitor_log, env_params) for i in range(n_envs)])
    if normalize:
        print("Normalizing input and return")
        env = VecNormalize(env, **normalize_kwargs)

# Optional Frame-stacking
n_stack = 1
if hyperparams.get('frame_stack', False):
    n_stack = hyperparams['frame_stack']
    env = VecFrameStack(env, n_stack)
    print("Stacking {} frames".format(n_stack))
    del hyperparams['frame_stack']

if args.save_video_interval != 0:
    env = VecVideoRecorder(env, save_path, record_video_trigger=lambda x: x % args.save_video_interval == 0,
                           video_length=args.save_video_length)

# Parse noise string for DDPG
if args.algo == 'ddpg' and hyperparams.get('noise_type') is not None:
    noise_type = hyperparams['noise_type'].strip()
    noise_std = hyperparams['noise_std']
    n_actions = env.action_space.shape[0]
    if 'adaptive-param' in noise_type:
        hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std, desired_action_stddev=noise_std)
    elif 'normal' in noise_type:
        hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    elif 'ornstein-uhlenbeck' in noise_type:
        hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                   sigma=noise_std * np.ones(n_actions))
    else:
        raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
    print("Applying {} noise with std {}".format(noise_type, noise_std))
    del hyperparams['noise_type']
    del hyperparams['noise_std']

if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
    # Continue training
    print("Loading pretrained agent")
    # Policy should not be changed
    del hyperparams['policy']

    model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                  tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

    exp_folder = args.trained_agent.split('.pkl')[0]
    if normalize:
        print("Loading saved running average")
        env.load_running_average(exp_folder)
else:
    # Train an agent from scratch
    model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

kwargs = {}
if args.log_interval > -1:
    kwargs = {'log_interval': args.log_interval}

model.learn(n_timesteps, **kwargs)

# Save trained model
print("Saving to {}".format(save_path))
model.save("{}/{}".format(save_path, env_id))

# Save hyperparams
with open(os.path.join(params_path, 'config.yml'), 'w') as f:
    saved_hyperparams.update(args.__dict__)
    saved_hyperparams.update(env_params)
    yaml.dump(saved_hyperparams, f)

if normalize:
    # Unwrap
    if isinstance(env, VecFrameStack):
        env = env.venv
    # Important: save the running average, for testing the agent we need that normalization
    env.save_running_average(params_path)

if not args.no_plot and n_timesteps > 1:
    results = load_group_results(save_path, verbose=True)
    f, _ = plot_results(results, average_group=True, shaded_std=False)
    f.savefig(os.path.join(save_path, 'results.png'), bbox_inches='tight', format='png')

if args.play > 0:
    test_path = os.path.join(save_path, 'test')
    hyperparams['normalize'] = normalize
    hyperparams['n_stack'] = n_stack
    if normalize:
        hyperparams['normalize_kwargs'] = normalize_kwargs

    env = create_test_env(env_id, n_envs=1, is_atari=is_atari,
                          stats_path=params_path, seed=0, log_dir=test_path, hyperparams=hyperparams)
    _ = env.reset()

    env = VecVideoRecorder(env, test_path,
                           record_video_trigger=lambda x: x == 0, video_length=args.play,
                           name_prefix="{}-{}-final".format(args.algo, env_id))

    obs = env.reset()
    for _ in range(args.play + 1):
        # action = [env.action_space.sample()]
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, _, _, _ = env.step(action)

    # Workaround for https://github.com/openai/gym/issues/893
    if n_envs == 1 and 'Bullet' not in env_id and not is_atari:
        env = env.venv
        # DummyVecEnv
        while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
            env = env.venv
        env.envs[0].env.close()
    else:
        # SubprocVecEnv
        env.close()

