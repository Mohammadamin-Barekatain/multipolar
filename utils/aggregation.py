"""Methods for aggregation
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX
"""

import tensorflow as tf


def variable_summaries(var, name_scope='summaries'):

    tf.summary.histogram(name_scope, var)

    shape = var.get_shape()
    if len(shape) == 2:
        for i in range(shape[-1]):
            tf.summary.scalar(name_scope+'_action'+str(i), tf.reduce_mean(var[:, i], axis=0))
    elif len(shape) == 3:
        for i in range(shape[-2]):
            for j in range(shape[-1]):
                tf.summary.scalar(name_scope+'_source'+str(i)+'_action_'+str(j), tf.reduce_mean(var[:, i, j], axis=0))
    else:
        raise Exception('full summary is not supported for the shape %s' % shape)


def get_aggregation_var(master_in, name_scope, n_sources, n_actions, no_bias, SDW, bias_layer_initializer=None,
                        summary=True):
    with tf.name_scope(name_scope):
        if SDW:
            # W = tf.layers.dense(master_in, n_sources * n_actions, activation=None)
            # W = tf.reshape(W, shape=[-1, n_sources, n_actions], name='scale')
            W = tf.ones([1, n_sources, n_actions], name='scale')
        else:
            W = tf.get_variable('scale', shape=[1, n_sources, n_actions],
                                dtype=tf.float32, trainable=True, initializer=tf.ones_initializer)

        if not no_bias:
            b = tf.layers.dense(master_in, n_actions, activation=None,
                                kernel_initializer=bias_layer_initializer, name='bias')
        else:
            # b = tf.zeros([1, n_actions], name='bias')
            b = tf.get_variable(name='bias', shape=[1, n_actions],
                                dtype=tf.float32, trainable=True, initializer=tf.zeros_initializer())

        if summary:
            variable_summaries(W, name_scope='W')
            variable_summaries(b, name_scope='b')

    assert W.get_shape()[1:] == (n_sources, n_actions)
    assert b.get_shape()[1:] == (n_actions,)

    return W, b


def affine_transformation(sources_actions, W, b, summary=True):

    mean_agg = tf.reduce_mean(sources_actions * W, axis=1, name='aggregated_actions')
    mean = tf.add(mean_agg, b, name='mean')

    if summary:
        tf.summary.histogram('mean', mean)
        tf.summary.histogram('mean_agg', mean_agg)

    return mean

