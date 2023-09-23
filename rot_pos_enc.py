import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint
from einops import repeat, rearrange

from .function import *





def rotate_half(tensor):
    t_shape = shape(tensor)
    t_1, t_2 = tensor[Ellipsis, 0::2], tensor[Ellipsis, 1::2]
    tensor = tf.reshape(tf.stack((-t_2, t_1), axis = -1), t_shape)
    return tensor



def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1]
    #cos_f, sin_f = map(lambda trif: tf.tile(trif(freqs)[None], [shape(t)[0], 1, 1]), (tf.cos, tf.sin))
    t_left, t, t_right = t[Ellipsis, :start_index], t[Ellipsis, start_index:end_index], t[Ellipsis, end_index:]
    t = (tf.einsum('ld,bl...d->bl...d', tf.cos(freqs), t) * scale) + (tf.einsum('ld,bl...d->bl...d', tf.sin(freqs), rotate_half(t)) * scale)
    return tf.concat((t_left, t, t_right), axis=-1)



def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = tf.einsum('...,f -> ...f', rotations, freq_ranges)
        rotations = tf.reshape(rotations, shape(rotations)[:-1] + [shape(rotations)[-1] * freq_ranges])

    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)



def rotary_embedding(
        tensor,
        length,
        features,
        custom_freqs=None,
        freq_type='lang',
        theta=10000,
        max_freq=10,
        freqs=1,
        trainable=False,
        scope='rot_emb'
        ):
    with tf.variable_scope(scope):
        # only for in-layer use
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freq_type == 'lang':
            freqs = 1. / (theta ** (np.arange(0, features, 2)[:(features // 2)].astype('float') / features))
        elif freq_type == 'pixel':
            freqs = np.linspace(1., max_freq / 2, features // 2) * np.pi
        elif freq_type == 'constant':
            freqs = np.ones(freqs, dtype=tensor.dtype)
        else:
            raise ValueError(f'unknown modality {freq_type}')

        freqs = tf.get_variable('frequency', [features // 2], dtype=tensor.dtype,
                initializer=tf.constant_initializer(freqs), trainable=trainable)

        freqs = tf.einsum('..., f -> ...f', tf.range(length, dtype=tensor.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        return apply_rotary_emb(freqs, tensor)


