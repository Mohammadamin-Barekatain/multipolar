"""callbacks for training a stable baseline model
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX

Parts of this script has been copied from https://github.com/hill-a/stable-baselines
"""

import numpy as np
from stable_baselines.results_plotter import ts2xy, load_results

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
