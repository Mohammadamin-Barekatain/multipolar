""" Wrappers for modifying the dynamics of OpenAI Gym environments.
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""

import gym
import inspect
import importlib


class ModifyEnvParams(gym.Wrapper):

    def __init__(self, env, **params):
        """
        Modify the parameters of the given Gym environment with params.

        env: (Gym Environment) the environment to wrap
        params: the parameters to change in env
        """
        gym.Wrapper.__init__(self, env)
        self.params = params

        # fid the path to the source code of the environment
        path = inspect.getfile(env.env.__class__)[:-3]
        path = path.split('/')
        idx = path.index('gym')
        path = path[idx:]
        path = ".".join(path)

        # import the environment
        imported_env = importlib.import_module(path)

        # change the parameters of the environment
        print('changing the environment parameters')
        for key, val in params.items():
            assert key in vars(imported_env), "{} is not a parameter in the env {}".format(key, imported_env)
            vars(imported_env)[key] = val
            print("{} changed to {}".format(key, val))

