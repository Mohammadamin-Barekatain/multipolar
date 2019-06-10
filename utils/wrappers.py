""" Wrappers for modifying the dynamics of OpenAI Gym environments.
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""
import os
import gym
import inspect
import importlib
import xml.etree.ElementTree as ET
from ast import literal_eval
import numpy as np
from roboschool import RoboschoolHopper
from gym.envs.box2d import LunarLander, BipedalWalker, CarRacing, BipedalWalkerHardcore


class ModifyBox2DEnvParams(gym.Wrapper):

    def __init__(self, env, **params):
        """
        Modify the parameters of the given Gym environment with params.

        env: (Gym Environment) the environment to wrap
        params: the parameters to change in env
        """
        gym.Wrapper.__init__(self, env)
        self.params = params

        # find the path to the source code of the environment
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


def get_string_from_tuple(s):
    s = str(s)
    s = s[1:-1]
    return s.replace(',', '')


def get_end(body):
    s = body.find('geom').attrib['fromto']
    s = s.replace(' ', ',')
    s = literal_eval(s)
    return s[-1]


def get_length(body):
    s = body.find('geom').attrib['fromto']
    s = s.replace(' ', ',')
    s = np.array(literal_eval(s))
    return np.linalg.norm(s[3:]-s[:3])


def set_body(body, start, end, pos=True):
    geom = body.find('geom')
    fromto = (0, 0, start, 0, 0, end)
    geom.set('fromto', get_string_from_tuple(fromto))

    if pos:
        joint = body.find('joint')
        pos = (0, 0, start)
        joint.set('pos', get_string_from_tuple(pos))


class ModifyHopperEnvParams(gym.Wrapper):

    def __init__(self, env, save_file, **params):
        """
        Modify the parameters of the given Roboschool Hopper environment with params.

        env: (Gym Environment) the environment to wrap
        params: the parameters to change in env
        """
        gym.Wrapper.__init__(self, env)
        self.params = params

        # find the path to the source code of the environment
        path = inspect.getfile(env.env.__class__)[:-3]
        idx = path.rfind('/')
        path = path[:idx]
        path = os.path.join(path, 'mujoco_assets/hopper.xml')

        # load xml file specifying hopper dynamics and Kinematics
        tree = ET.parse(path)
        root = tree.getroot()
        body_parts = {body.attrib['name']: body for body in root.iter('body')}

        # change the parameters of the environment
        print('changing the environment parameters')
        for key, val in params.items():
            val = float(val)
            if 'length' in key:
                key = key.split('_')[0]
                change = False
                for body_name in ['leg', 'thigh', 'torso']:
                    change = change or key == body_name
                    if change:
                        if key == body_name:
                            end = get_end(body_parts[body_name])
                            start = end + val
                            print("length of {} changed to {}".format(body_name, val))
                        else:
                            start = end + get_length(body_parts[body_name])
                        set_body(body_parts[body_name], start, end, pos=(body_name != 'torso'))
                        end = start

                if key == 'foot':
                    foot_fromto = (-val/3.0, 0, 0.1, val/3.0*2, 0, 0.1)
                    body_parts[key].find('geom').set('fromto', get_string_from_tuple(foot_fromto))
                    print("length of foot changed to {}".format(val))

                elif not change:
                    raise ValueError('hopper has no body part named {}'.format(key))

            elif key == 'size':
                for name, body in body_parts.items():
                    geom = body.find('geom')
                    size = float(geom.attrib['size'])
                    size = size * val
                    geom.set('size', str(size))
                    print("size of {} changed to {}".format(name, size))

            elif key == 'damping':
                joint = root.find('default').find('joint')
                joint.set('damping', str(val))
                print("damping changed to {}".format(val))

            else:
                raise ValueError('{} is either not a parameter in the env {} or not supported'.format(key, env))

        tree.write(save_file)


def modify_env_params(env, params_path=None, **params):

    if isinstance(env.env, LunarLander) or isinstance(env.env, BipedalWalker) \
            or isinstance(env.env, CarRacing) or isinstance(env.env, BipedalWalkerHardcore):
        return ModifyBox2DEnvParams(env, **params)

    if isinstance(env.env, RoboschoolHopper):
        assert params_path is not None, "params_path must be provided for modifying Hopper"
        save_file = os.path.join(params_path, "Hopper.xml")
        env = ModifyHopperEnvParams(env, save_file, **params)
        if hasattr(env.env, 'model_xml'):
            env.env.model_xml = '/' + save_file
        elif hasattr(env.env.env, 'model_xml'):
            env.env.env.model_xml = '/' + save_file
        else:
            raise ValueError("cannot change the env, please check the source code")
        return env

    else:
        raise ValueError("Modifying environment parameters is not supported for {}".format(env.env))
