""" Utility functions.
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX

some functions has been copied from: https://github.com/openai/baselines and https://github.com/araffin/rl-baselines-zoo
"""

import time
import difflib
import os
import inspect
import glob
import yaml
import pandas
import os.path as osp
import gym
from collections import namedtuple

from gym.envs.registration import load
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.policies import FeedForwardPolicy as BasePolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.bench import Monitor
from stable_baselines import logger
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, DDPG
from stable_baselines.results_plotter import load_results, ts2xy
from utils.wrappers import ModifyEnvParams

# Temp fix until SAC is integrated into stable_baselines
try:
    from stable_baselines import SAC
except ImportError:
    SAC = None
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack, SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'sac': SAC,
    'ppo2': PPO2
}


# ================== Custom Policies =================

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=[64],
                                              layer_norm=True,
                                              feature_extraction="mlp")


class CustomMlpPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs,
                                              layers=[16],
                                              feature_extraction="mlp")


if SAC is not None:
    from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy


    class CustomSACPolicy(SACPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                                  layers=[256, 256],
                                                  feature_extraction="mlp")


    register_policy('CustomSACPolicy', CustomSACPolicy)

register_policy('CustomDQNPolicy', CustomDQNPolicy)
register_policy('CustomMlpPolicy', CustomMlpPolicy)


def make_env(env_id, rank=0, seed=0, log_dir=None, env_params=[]):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    """
    if log_dir is None and log_dir != '':
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        set_global_seeds(seed + rank)
        env = gym.make(env_id)
        if len(env_params) > 0:
            env = ModifyEnvParams(env, **env_params)
        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env

    return _init


def create_test_env(env_id, n_envs=1, stats_path=None, seed=0, log_dir=None, should_render=True, hyperparams=None,
                    env_params={}):
    """
    Create environment for testing a trained agent

    :param env_id: (str)
    :param n_envs: (int) number of processes
    :param stats_path: (str) path to folder containing saved running averaged
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param should_render: (bool) For Pybullet env, display the GUI
    :param hyperparams: (dict) Additional hyperparams for the env (ex: n_stack)
    :param env_params: (dict) the parameters to change in env
    :return: (gym.Env)
    """
    # If the environment is not found, suggest the closest match
    registered_envs = set(gym.envs.registry.env_specs.keys())
    if env_id not in registered_envs:
        closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    is_atari = 'NoFrameskip' in env_id

    # HACK to save logs
    if log_dir is not None:
        os.environ["OPENAI_LOG_FORMAT"] = 'log'
        os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        logger.configure()

    # Create the environment and wrap it if necessary
    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif n_envs > 1:
        env = SubprocVecEnv([make_env(env_id, i, seed, log_dir, env_params) for i in range(n_envs)])
    # Pybullet envs does not follow gym.render() interface
    elif "Bullet" in env_id:
        spec = gym.envs.registry.env_specs[env_id]
        class_ = load(spec._entry_point)
        # HACK: force SubprocVecEnv for Bullet env that does not
        # have a render argument
        render_name = None
        use_subproc = 'renders' not in inspect.getfullargspec(class_.__init__).args
        if not use_subproc:
            render_name = 'renders'
        # Dev branch of pybullet
        # use_subproc = use_subproc and 'render' not in inspect.getfullargspec(class_.__init__).args
        # if not use_subproc and render_name is None:
        #     render_name = 'render'

        # Create the env, with the original kwargs, and the new ones overriding them if needed
        def _init():
            # TODO: fix for pybullet locomotion envs
            env = class_(**{**spec._kwargs}, **{render_name: should_render})
            if len(env_params) > 0:
                env = ModifyEnvParams(env, **env_params)
            env.seed(0)
            if log_dir is not None:
                env = Monitor(env, os.path.join(log_dir, "0"), allow_early_resets=True)
            return env

        if use_subproc:
            env = SubprocVecEnv([make_env(env_id, 0, seed, log_dir, env_params)])
        else:
            env = DummyVecEnv([_init])
    else:
        env = DummyVecEnv([make_env(env_id, 0, seed, log_dir, env_params)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams['normalize']:
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            env = VecNormalize(env, training=False, **hyperparams['normalize_kwargs'])
            env.load_running_average(stats_path)

        n_stack = hyperparams.get('n_stack', 0)
        if n_stack > 0:
            print("Stacking {} frames".format(n_stack))
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def get_trained_models(log_folder):
    """
    :param log_folder: (str) Root log folder
    :return: (dict) Dict representing the trained agent
    """
    algos = os.listdir(log_folder)
    trained_models = {}
    for algo in algos:
        for env_id in glob.glob('{}/{}/*.pkl'.format(log_folder, algo)):
            # Retrieve env name
            env_id = env_id.split('/')[-1].split('.pkl')[0]
            trained_models['{}-{}'.format(algo, env_id)] = (algo, env_id)
    return trained_models


def get_latest_run_id(log_path, env_id, exp_name=None):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    if exp_name:
        loc = log_path + "/{}_{}_[0-9]*".format(env_id, exp_name)
    else:
        loc = log_path + "/{}_[0-9]*".format(env_id)

    for path in glob.glob(loc):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[0:1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)

    return max_run_id


def get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False):
    """
    :param stats_path: (str)
    :param norm_reward: (bool)
    :param test_mode: (bool)
    :return: (dict, str)
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, 'config.yml')
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, 'config.yml'), 'r') as f:
                hyperparams = yaml.load(f)
            hyperparams['normalize'] = hyperparams.get('normalize', False)
        else:
            obs_rms_path = os.path.join(stats_path, 'obs_rms.pkl')
            hyperparams['normalize'] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams['normalize']:
            if isinstance(hyperparams['normalize'], str):
                normalize_kwargs = eval(hyperparams['normalize'])
                if test_mode:
                    normalize_kwargs['norm_reward'] = norm_reward
            else:
                normalize_kwargs = {'norm_obs': hyperparams['normalize'], 'norm_reward': norm_reward}
            hyperparams['normalize_kwargs'] = normalize_kwargs
    return hyperparams, stats_path





Result_obj = namedtuple('Result', 'monitor dirname')
Result_obj.__new__.__defaults__ = (None,) * len(Result_obj._fields)

def load_group_results(root_dir_or_dirs, env='', verbose=False, mask=None):
    '''
    load summaries of runs from a list of directories (including subdirectories)
    Arguments:
    verbose: bool - if True, will print out list of directories from which the data is loaded. Default: False
    env: if provided, only results of the corresponding env would be loaded.
    mask: if provided,
    Returns:
    List of Result objects with the following fields:
         - dirname - path to the directory data was loaded from
         - monitor - if enable_monitor is True, this field contains pandas dataframe with loaded monitor.csv file
         (or aggregate of all *.monitor.csv files in the directory)
    '''
    import re
    if isinstance(root_dir_or_dirs, str):
        rootdirs = [osp.expanduser(root_dir_or_dirs)]
    else:
        rootdirs = [osp.expanduser(d) for d in root_dir_or_dirs]
    allresults = []
    for rootdir in rootdirs:
        assert osp.exists(rootdir), "%s doesn't exist"%rootdir
        for dirname, dirs, files in os.walk(rootdir):
            if '-proc' in dirname:
                files[:] = []
                continue
            monitor_re = re.compile(r'(\d+\.)?(\d+\.)?monitor\.csv')
            if set(['monitor.json']).intersection(files) or \
               any([f for f in files if monitor_re.match(f)]):  # also match monitor files like 0.1.monitor.csv
                # used to be uncommented, which means do not go deeper than current directory if any of the data files
                # are found
                # dirs[:] = []
                result = {'dirname': dirname}

                try:
                    name = dirname.split('/')[-1]
                    if not name.startswith(env) or (mask and (not re.search(mask, dirname))) or dirname.count('/') != 2:
                        continue

                    data = load_results(dirname)
                    if len(data) < 2:
                        print('empty dir ', dirname)
                        continue
                    result['monitor'] = pandas.DataFrame(load_results(dirname))
                # except Monitor.LoadMonitorResultsError:
                #     print('skipping %s: no monitor files' % dirname)
                except Exception as e:
                    print('exception loading monitor file in %s: %s' % (dirname, e))

                if result.get('monitor') is not None:
                    allresults.append(Result_obj(**result))
                    if verbose:
                        print('successfully loaded %s' % dirname)

    if verbose:
        print('loaded %i results' % len(allresults))
    return allresults


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    currently supports integers, strings, boolean and None
    """
    def convert(value):
        if value.isdigit():
            return float(value)
        if value in ['True', 'False']:
            return bool(value)
        if value == 'None':
            return None
        return value

    retval = {}
    preceded_by_key = False

    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = convert(value)
            else:
                key = arg[2:]
                preceded_by_key = True

        elif preceded_by_key:
            retval[key] = convert(arg)
            preceded_by_key = False
        else:
            raise ValueError("Invalid arg: %s" % arg)

    return retval
