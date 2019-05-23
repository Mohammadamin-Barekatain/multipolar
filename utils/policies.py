"""Policies for RL algorithms
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX

Small parts of this script has been copied from https://github.com/hill-a/stable-baselines
"""

from stable_baselines.common import tf_util
import tensorflow as tf
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, DDPG, SAC
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import nature_cnn
from stable_baselines.common.policies import register_policy
from stable_baselines.common.policies import FeedForwardPolicy as BasePolicy
from stable_baselines.deepq.policies import FeedForwardPolicy as DQNFeedForwardPolicy
from stable_baselines.sac.policies import FeedForwardPolicy as SACFeedForwardPolicy
from stable_baselines.sac.policies import mlp, gaussian_likelihood, gaussian_entropy, apply_squashing_func
from stable_baselines.a2c.utils import find_trainable_variables
import numpy as np

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
    'ppo2': PPO2
}


class CustomDQNPolicy(DQNFeedForwardPolicy):
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


class CustomSACPolicy(SACFeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              feature_extraction="mlp")


class AggregatePolicy(SACFeedForwardPolicy):
    """
    Policy object that implements a DDPG-like actor critic, by aggregating a set of source policies using a master model.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the master model (if None, state-independent master model)
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for source policies path and the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", layer_norm=False, act_fun=tf.nn.relu, **kwargs):

        assert 'source_policy_paths' in kwargs, 'path to source policies is not provided.'
        source_policy_paths = kwargs['source_policy_paths']
        del kwargs['source_policy_paths']

        super(AggregatePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                              layers=layers, cnn_extractor=cnn_extractor, layer_norm=layer_norm,
                                              feature_extraction=feature_extraction, act_fun=act_fun, **kwargs)

        self.K = len(source_policy_paths)
        self.D = self.ac_space.shape[0]
        self.n_batch = n_batch
        # override layers
        if layers is None:
            self.layers = []

        sources_actions = []
        for ind, path in enumerate(source_policy_paths):
            # load the model
            algo = path.split('/')[1]
            model = ALGOS[algo].load(path, verbose=1)

            def predict(obs):
                return model.policy_tf.step(obs, deterministic=True)
            action = tf.py_func(predict, [self.obs_ph], tf.float32, name='source_action' + str(ind))

            action.set_shape((self.n_batch, self.D))
            action = tf.stop_gradient(action)

            sources_actions.append(action)

        sources_actions = tf.stack(sources_actions)  # shape = K x batch x D
        sources_actions = tf.transpose(sources_actions, perm=[1, 0, 2], name='source_actions')  # shape = batch x K x D

        assert sources_actions.get_shape()[1:] == (self.K, self.D)
        self.sources_actions = sources_actions

    def get_aggregation_var(self, pi_h, reuse, scope, bias_size):
        with tf.variable_scope(scope, reuse=reuse):
            if pi_h is not None:
                pi_h = tf.layers.dense(pi_h, self.D * self.K + bias_size, activation=None)
                W, b = tf.split(pi_h, [self.D * self.K, bias_size], -1)

                b = tf.identity(b, name='bias')
                W = tf.reshape(W, shape=[-1, self.K, self.D], name='scale')

            else:
                b = tf.get_variable('bias', shape=[1, bias_size],
                                    dtype=tf.float32, trainable=True, initializer=tf.zeros_initializer)
                W = tf.get_variable('scale', shape=[1, self.K, self.D],
                                    dtype=tf.float32, trainable=True, initializer=tf.ones_initializer)

        assert W.get_shape()[1:] == (self.K, self.D)
        assert b.get_shape()[1:] == (bias_size,)

        return W, b

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            pi_h = None
            if len(self.layers) > 0 or self.feature_extraction == "cnn":
                if self.feature_extraction == "cnn":
                    pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
                else:
                    pi_h = tf.layers.flatten(obs)
                pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            mu_W, mu_b = self.get_aggregation_var(pi_h, reuse, scope='master_mu', bias_size=self.D)

            self.act_mu = mu_ = tf.reduce_mean(self.sources_actions * mu_W, axis=1) + mu_b
            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            sd_W, sd_b = self.get_aggregation_var(pi_h, reuse, scope='master_sd', bias_size=self.D)
            log_std = tf.reduce_mean(self.sources_actions * sd_W, axis=1) + sd_b

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
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    # def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn", create_vf=True, create_qf=True):
    #     if obs is None:
    #         obs = self.processed_obs
    #
    #     with tf.variable_scope(scope, reuse=reuse):
    #
    #         if self.feature_extraction == "cnn":
    #             critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
    #         else:
    #             critics_h = tf.layers.flatten(obs)
    #
    #         if create_vf:
    #             # Value function
    #             with tf.variable_scope('vf', reuse=reuse):
    #                 vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
    #                 vf_W, vf_b = self.get_aggregation_var(vf_h, reuse, scope='master', bias_size=1)
    #                 value_fn = tf.reduce_mean(tf.reduce_mean(self.sources_actions * vf_W, axis=1) + vf_b,
    #                                           axis=-1, name='vf')
    #             self.value_fn = value_fn
    #
    #         if create_qf:
    #             # Concatenate preprocessed state and action
    #             qf_h = tf.concat([critics_h, action], axis=-1)
    #
    #             # Double Q values to reduce overestimation
    #             with tf.variable_scope('qf1', reuse=reuse):
    #                 qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
    #                 qf1_W, qf1_b = self.get_aggregation_var(qf1_h, reuse, scope='master', bias_size=1)
    #                 qf1 = tf.reduce_mean(tf.reduce_mean(self.sources_actions * qf1_W, axis=1) + qf1_b,
    #                                      axis=-1, name="qf1")
    #
    #             with tf.variable_scope('qf2', reuse=reuse):
    #                 qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
    #                 qf2_W, qf2_b = self.get_aggregation_var(qf2_h, reuse, scope='master', bias_size=1)
    #                 qf2 = tf.reduce_mean(tf.reduce_mean(self.sources_actions * qf2_W, axis=1) + qf2_b,
    #                                      axis=-1, name="qf2")
    #
    #             self.qf1 = qf1
    #             self.qf2 = qf2
    #
    #     return self.qf1, self.qf2, self.value_fn


class StateIndependentAggregatePolicy(AggregatePolicy):
    """
    Policy object that implements actor critic aggregation where aggregation in actor is state-independent

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

        super(StateIndependentAggregatePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                              feature_extraction="mlp", **_kwargs)


register_policy('StateIndependentAggregatePolicy', StateIndependentAggregatePolicy)
register_policy('CustomSACPolicy', CustomSACPolicy)
register_policy('CustomDQNPolicy', CustomDQNPolicy)
register_policy('CustomMlpPolicy', CustomMlpPolicy)
