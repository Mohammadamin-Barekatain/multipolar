"""Distributions used in RL algorithms
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX

Parts of this script has been copied from https://github.com/hill-a/stable-baselines
"""

import tensorflow as tf
from gym import spaces
from stable_baselines.a2c.utils import linear, ortho_init
from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution, ProbabilityDistributionType

from .aggregation import get_aggregation_var, affine_transformation


# ToDo: implement CategoricalProbabilityDistributionType, MultiCategoricalProbabilityDistributionType,
#  BernoulliProbabilityDistributionType


class DiagGaussianProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size, sources_actions, no_bias, SDW, summary):
        """
        The probability distribution type for multivariate gaussian input

        :param size: (int) the number of dimensions of the multivariate gaussian
        :param sources_actions: (tensor) actions of source policies in shape [n_batch, n_sources, n_actions]
        :param no_bias: (bool) weather to use bias in aggregation
        """
        self.size = size
        self.sources_actions = sources_actions
        self.no_bias = no_bias
        self.summary = summary
        self.SDW=SDW

    def probability_distribution_class(self):
        return DiagGaussianProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):

        master_W, master_b = get_aggregation_var(pi_latent_vector, 'master', self.sources_actions.get_shape()[1],
                                                 self.sources_actions.get_shape()[2], no_bias=self.no_bias,
                                                 SDW=self.SDW, bias_layer_initializer=ortho_init(init_scale),
                                                 summary=self.summary)
        mean = affine_transformation(self.sources_actions, master_W, master_b, summary=self.summary)

        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)

        return self.proba_distribution_from_flat(pdparam), mean, q_values

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


def make_mlap_proba_dist_type(ac_space, sources_actions, no_bias, SDW, summary):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :param size: (int) the number of dimensions of the multivariate gaussian
    :param sources_actions: (tensor) actions of source policies in shape [n_batch, n_sources, n_actions]
    :param no_bias: (bool) weather to use bias in aggregation
    :return: (ProbabilityDistributionType) the approriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        return DiagGaussianProbabilityDistributionType(ac_space.shape[0], sources_actions, no_bias, SDW, summary)
    elif isinstance(ac_space, spaces.Discrete):
        raise NotImplementedError('probability distribution, not implemented for Discrete action space')
        # return CategoricalProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        raise NotImplementedError('probability distribution, not implemented for MultiDiscrete action space')
        # return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        raise NotImplementedError('probability distribution, not implemented for MultiBinary action space')
        # return BernoulliProbabilityDistributionType(ac_space.n)
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(ac_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")

