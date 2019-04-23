"""callbacks for training a stable baseline model
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX

Parts of this script has been copied from https://github.com/hill-a/stable-baselines
"""

import numpy as np
import os
import gym
from utils import create_test_env
from stable_baselines.results_plotter import ts2xy, load_results
from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

class Callback(object):
    """Abstract base class used to build new callbacks."""

    def callback(self, _locals, _globals):
        """
            Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
            :param _locals: (dict)
            :param _globals: (dict)
            Returning False will stop training early
        """


class ModelCheckpoint(Callback):
    """"Save the model after some epoch."""
    #ToDo: complete the doc and save every epoch. add a flag for only saving the best like:
    # https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L633

    def __init__(self, log_dir, interval=1000, xaxis='timesteps'):
        self.best_mean_reward = -np.inf
        self.n_steps = 0
        self.log_dir = log_dir
        self.xaxis = xaxis
        self.interval = interval

    def callback(self, _locals, _globals):
        # Print stats every 'interval' calls
        if (self.n_steps + 1) % self.interval == 0:
            # Evaluate policy performance
            x, y = ts2xy(load_results(self.log_dir), self.xaxis)
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], self.xaxis)
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(self.log_dir + 'best_model.pkl')
        self.n_steps += 1
        return True

class VideoRecorder(Callback):
    """"Save the video of the agent after some epoch."""
    def __init__(self, env_id, video_folder, hyperparams, params_path, is_atari, video_length, deterministic=True, name_prefix='video', interval=10000):
        """
        :param env_id: environment id
        :param video_folder: (str) Where to save videos
        :param video_length: (int)  Length of recorded videos
        :param name_prefix: (str) Prefix to the video name
        :param interval:
        """
        #ToDo: finish the doc string

        self.best_mean_reward = -np.inf
        self.n_steps = 0
        self.video_folder = video_folder
        self.interval = interval
        self.video_length = video_length
        self.deterministic = deterministic

        test_path = os.path.join(video_folder, 'video')

        env = create_test_env(env_id, n_envs=1, is_atari=is_atari,
                              stats_path=params_path, seed=0, hyperparams=hyperparams)
        env.reset()

        self.env = VecVideoRecorder(env, test_path,
                               record_video_trigger=lambda x: x == 0, video_length=video_length,
                               name_prefix=name_prefix)


    def callback(self, _locals, _globals):
        # record every 'interval' calls
        if self.n_steps % self.interval == 0:
            env = self.env

            obs = env.reset()
            for _ in range(self.video_length + 1):
                # action = [env.action_space.sample()]
                action, _ = _locals['self'].predict(obs, deterministic=self.deterministic)
                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, _, _, _ = env.step(action)

        self.n_steps += 1
        return True