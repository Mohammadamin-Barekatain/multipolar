"""Policies for RL algorithms
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX

Small parts of this script has been copied from https://github.com/hill-a/stable-baselines
"""

from stable_baselines.common import tf_util
import tensorflow as tf
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, DDPG, SAC
from stable_baselines.common.policies import nature_cnn, register_policy
from stable_baselines.deepq.policies import FeedForwardPolicy as DQNFeedForwardPolicy
from stable_baselines.sac.policies import FeedForwardPolicy as SACFeedForwardPolicy
from stable_baselines.common.policies import FeedForwardPolicy, ActorCriticPolicy, mlp_extractor
from stable_baselines.sac.policies import mlp, gaussian_likelihood, gaussian_entropy, apply_squashing_func
from stable_baselines.a2c.utils import linear
from .distributions import make_mlap_proba_dist_type
import warnings
import gym
from .aggregation import get_aggregation_var, affine_transformation
import numpy as np
from gym import spaces


EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'sac': SAC,
    'ppo2': PPO2,
    'mlap-ppo2': PPO2,
    'mlap-sac': SAC
}


def get_predict_func(path, ac_space):
    # load the model
    algo = path.split('/')[1].split('_')[0]
    model = ALGOS[algo].load(path, verbose=1)

    def _predict(obs):
        action, _ = model.predict(obs, deterministic=True)

        if isinstance(ac_space, gym.spaces.Box):
            action = np.clip(action, ac_space.low + EPS, ac_space.high - EPS)

        return action

    return _predict


def get_sources_actions(obs_ph, source_policy_paths, n_batch, n_actions, ac_space, action_dtype=tf.float32):
    sources_actions = []
    for ind, path in enumerate(source_policy_paths):

        predict = get_predict_func(path, ac_space)

        action = tf.py_func(predict, [obs_ph], action_dtype, name='source_actions' + str(ind))
        if action_dtype != tf.float32:
            action = tf.one_hot(action, n_actions, dtype=tf.float32)

        action.set_shape((n_batch, n_actions))
        action = tf.stop_gradient(action)

        sources_actions.append(action)

    sources_actions = tf.stack(sources_actions)  # shape = K x batch x D
    sources_actions = tf.transpose(sources_actions, perm=[1, 0, 2], name='sources_actions')  # shape = batch x K x D

    assert sources_actions.get_shape()[1:] == (len(source_policy_paths), n_actions)

    return sources_actions


def get_master_config(kwargs):

    assert 'source_policy_paths' in kwargs, 'path to source policies is not provided.'
    assert 'SDW' in kwargs, 'state dependency of scales is not specified'
    assert 'no_bias' in kwargs, 'inclusion of bias is not specified'
    assert isinstance(kwargs['no_bias'], bool), '\'no_bias\' must be bool'
    assert isinstance(kwargs['SDW'], bool), '\'SDW\' must be bool'

    source_policy_paths = kwargs['source_policy_paths']
    SDW = kwargs['SDW']
    no_bias = kwargs['no_bias']

    del kwargs['source_policy_paths']
    del kwargs['SDW']
    del kwargs['no_bias']

    return source_policy_paths, SDW, no_bias


class CustomDQNPolicy(DQNFeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=[64],
                                              layer_norm=True,
                                              feature_extraction="mlp")


class CustomMlpPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs,
                                              layers=[16],
                                              feature_extraction="mlp")


class CustomSACPolicy(SACFeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              feature_extraction="mlp")


class AggregatePolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, by aggregating a set of source policies using a master model.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the master model
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture  for master model
        (see mlp_extractordocumentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for source policies path and the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):

        source_policy_paths, SDW, no_bias = get_master_config(kwargs)

        super(AggregatePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                              scale=(feature_extraction == "cnn"))

        if isinstance(ac_space, spaces.Box):
            n_actions = self.ac_space.shape[0]
            action_dtype = tf.float32

        elif isinstance(ac_space, spaces.Discrete):
            n_actions = ac_space.n
            action_dtype = tf.int64

        else:
            raise NotImplementedError("MLAP is not implemented for the required action space")

        sources_actions = get_sources_actions(self.obs_ph, source_policy_paths, n_batch,
                                              n_actions, ac_space, action_dtype)
        self.pdtype = make_mlap_proba_dist_type(ac_space, sources_actions, no_bias, SDW, summary=reuse)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self.value_fn = linear(vf_latent, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})


class SACAggregatePolicy(SACFeedForwardPolicy):
    """
    Policy object that implements a DDPG-like actor critic, by aggregating a set of source policies using a master model.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the master model
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for source policies path and the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", layer_norm=False, act_fun=tf.nn.relu, **kwargs):

        source_policy_paths, self.SDW, self.no_bias = get_master_config(kwargs)

        super(SACAggregatePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                 layers=layers if layers is None or len(layers) > 0 else None,
                                                 cnn_extractor=cnn_extractor, layer_norm=layer_norm,
                                                 feature_extraction=feature_extraction, act_fun=act_fun, **kwargs)
        if layers is not None and len(layers) == 0:
            self.layers = layers

        self.n_sources = len(source_policy_paths)
        self.n_actions = self.ac_space.shape[0]
        self.sources_actions = get_sources_actions(self.obs_ph, source_policy_paths, n_batch, self.n_actions, ac_space)

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            if len(self.layers) > 0:
                pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            master_W, master_b = get_aggregation_var(pi_h, name_scope='master', n_sources=self.n_sources,
                                                     SDW=self.SDW, n_actions=self.n_actions, no_bias=self.no_bias)

            self.act_mu = mu_ = affine_transformation(self.sources_actions, master_W, master_b)

            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None, name='log_std')

        # OpenAI Variation to cap the standard deviation
        # activation = tf.tanh # for log_std
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probabilty
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)

        if isinstance(self.ac_space, gym.spaces.Box):
            policy = tf.clip_by_value(policy, self.ac_space.low + EPS, self.ac_space.high - EPS)
            deterministic_policy = tf.clip_by_value(deterministic_policy, self.ac_space.low + EPS, self.ac_space.high - EPS)

        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi


# ToDo: rename this to offPolicy
class SACTwoLayerMlpAggregatePolicy(SACAggregatePolicy):
    """
    Policy object that implements DDPG-like actor critic aggregation, using a MLP (2 layers of 64) in master model

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for source policies path and the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):

        super(SACTwoLayerMlpAggregatePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                            reuse, feature_extraction="mlp", **_kwargs)


class MlpAggregatePolicy(AggregatePolicy):
    """
    Policy object that implements actor critic aggregation, using a MLP (2 layers of 64) in master model

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpAggregatePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                 feature_extraction="mlp", **_kwargs)


register_policy('SACTwoLayerMlpAggregatePolicy', SACTwoLayerMlpAggregatePolicy)
register_policy('MlpAggregatePolicy', MlpAggregatePolicy)
register_policy('CustomSACPolicy', CustomSACPolicy)
register_policy('CustomDQNPolicy', CustomDQNPolicy)
register_policy('CustomMlpPolicy', CustomMlpPolicy)
