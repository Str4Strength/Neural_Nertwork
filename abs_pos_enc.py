import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *


def pos_enc_1d(
        tensor,
        mask=None
        ):
    dtype = tensor.dtype
    B, L, C = shape(tensor)
    if C%2 != 0: return tf.zeros_like(tensor)

    div = tf.math.exp(tf.range(start=0, limit=C, delta=2, dtype=dtype)*
        -(tf.math.log(10000.)/C))

    d_pos = tf.range(L, dtype=dtype)[:, None] * div[None]

    pe = tf.stack([tf.math.sin(d_pos), tf.math.cos(d_pos)], axis=-1) # L, C//2, 2
    abs_pe = tf.reshape(pe, [L, C])
    pos_enc = abs_pe[None] # 1, L, C
    if mask is not None: pos_enc*=mask
    return pos_enc


def pos_enc_2d(
        tensor,
        mask=None
        ):
    dtype = tensor.dtype
    B, H, W, C = shape(tensor)
    if C%4 != 0: return tf.zeros_like(tensor)

    div = tf.math.exp(tf.range(start=0, limit=C//2, delta=2, dtype=dtype)*
        -(tf.math.log(10000.)/(C//2)))

    d_pos_h = tf.range(H, dtype=dtype)[:, None] * div[None]
    d_pos_w = tf.range(W, dtype=dtype)[:, None] * div[None]

    pe_h = tf.stack([tf.math.sin(d_pos_h), tf.math.cos(d_pos_h)], axis=-1) # H, C//4, 2
    pe_w = tf.stack([tf.math.sin(d_pos_w), tf.math.cos(d_pos_w)], axis=-1) # W, C//4, 2
    abs_pe = tf.concat([tf.tile(pe_w[None], [H, 1, 1, 1]),
        tf.tile(pe_h[:, None], [1, W, 1, 1])], axis=-1)
    abs_pe = tf.reshape(abs_pe, [H, W, C])
    pos_enc = abs_pe[None] # 1, H, W, C
    if mask is not None: pos_enc*=mask
    return pos_enc



