import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint





def sine_spe(
        tensor,
        heads = 8,
        realizations = 8,
        sines = 1
        trainable = True,
        scope = 'sine_spe'
        ):
    with tf.variable_scope(scope):
        t_shape = shape(tensor)[-1]
        freqs = tf.get_variable('freqs', (heads, t_shape[-1], sines), dtype=tensor.dtype,
                initialzier=tf.random_normal_initializer, trainable=trainable)
        offsets = tf.get_variable('offsets', (heads, t_shape[-1], sines), dtype=tensor.dtype,
                initialzier=tf.random_normal_initializer, trainable=trainable)
        gains = np.random.normal('gains', (heads, t_shape[-1], sines), dtype=tensor.dtype,
                initialzier=tf.random_normal_initializer, trainable=trainable)

        max_len = t_shape[1]
        indices = tf.cast(tf.range(0, max_len), dtype=tensor.dtype)
        freqs = tf.sigmoid(freqs[:, :, None]) / 2.

        phases_q = 2 * np.pi * freqs * indices[None, None, :, None] + offsets[:, :, None]
        omega_q = tf.reshape(tf.stack([tf.cos(phases_q), tf.sin(phases_q)], axis=-1),
                [1, heads, t_shape[-1], max_len, 2 * sines])

        phases_k = 2 * np.pi * freqs * indices[None, None, :, None]
        omega_k = tf.reshape(tf.stack([tf.cos(phases_k), tf.sin(phases_k)], axis=-1),
                [1, heads, t_shape[-1], max_len, 2 * sines])

        gains = tf.softplus(gains)
        gains = tf.reshape(tf.tile(gains[Ellipsis, None], [1, 1, 1, 2]). [*shape(gains)[:-1], 2 * sines])

        z = tf.random.normal((1, heads, t_shape[-1], 2 * sines, realizations)) * tf.rsqrt(2 * sines)
        z *= gains[None, Ellipsis, None]

        qbar = tf.einsum('ohdls,ohdsr->ohdlr', omega_q, z)
        kbar = tf.einsum('ohdls,ohdsr->ohdlr', omega_k, z)

        qbar = tf.transpose(qbar, [0, 3, 1, 2, 4])
        kbar = tf.tranapose(kbar, [0, 3, 1, 2, 4])

        scale = float(realizations * t_shape[-1]) ** 0.25

        return qbar / scale, kbar / scale



def conv_spe(
        tensor,
        features,
        heads = 8,
        realizations = 8,
        kernels = 200,
        trainable = True,
        scope = 'conv_spe'
        ):
    with tf.variable_scope(scope):
        if isinstance(kernels, int): kernels = (kernels,) * rank
        t_shape = shape(tensor)     # b, ... , c
        rank = len(t_shape[1:-1])

        # larger shape to avoid border effects
        s_shape = [4 * k + s for (k, s) in zip(kernels, t_shape[1:-1])]
        z = tf.random.normal((realizations, *s_shape[1:-1], t_shape[-1]))

        t_bar = convolution(z, rank, heads * features, kernels, strides=1, padding='valid', bias=False,
                groups=heads * features, trainable=trainable, scope=f'conv{rank}d')

        indices = [slice(realizations), *[slice(k, k + s, 1) for (k, s) in list(zip(kernels, t_shape[1:-1]))], slice(heads * features)]

        t_bar = tf.reshape(t_bar[indices], [1, realizations, *t_shape[1:-1], heads, features])
        t_bar = tf.transpose(t_bar, [0, *list(range(2, rank + 2)), rank + 2, rank + 3, 1])

        scale = (realizations * features) ** 0.25

    return t_bar / scale




