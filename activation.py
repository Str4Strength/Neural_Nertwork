import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def sigmoid(x, mask=None):
    x = tf.sigmoid(x)
    if exists(mask): x *= mask
    return x, mask

def gelu(x):
    cdf = .5 * (1. + tf.math.tanh(np.sqrt(2./np.pi) * (x + .044715 * tf.math.pow(x, 3.))))
    return x * cdf

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

