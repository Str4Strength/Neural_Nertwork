import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

from .function import *



# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def weight_(
        shape,
        dtype,
        init=None,
        gain=1,
        use_wscale=False,
        lrmul=1,
        function=None,
        trainable=True,
        scope='weight',
        ):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)

    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    if init is None: init = tf.initializers.random_normal(0, init_std)
    weight = tf.get_variable(scope, shape=shape, dtype=dtype, initializer=init, trainable=trainable) * runtime_coef

    if exists(function):
        if isinstance(function, list) or isinstance(function, set):
            for f in function:
                if exists(f): weight = f(weight)
        else:
            weight = function(weight)

    return weight



def bias_(
        shape,
        dtype,
        init=None,
        lrmul=1,
        function=None,
        trainable=True,
        scope='bias',
        ):
    if init is None: init = tf.zeros_initializer
    bias = tf.get_variable(scope, shape=shape, dtype=dtype, initializer=init, trainable=trainable) * lrmul

    if function is not None:
        if isinstance(function, list) or isinstance(function, set):
            for f in function: bias = f(bias)
        else:
            bias = function(bias)

    return bias



def quantization_noise(
        weight,
        in_features,
        out_features,
        p,
        block_size,
        ):
    if p <= 0: return weight

    #in_features = (in_features,) if isinstance(in_features, int) else tuple(in_features)
    #out_features = (out_features,) if isinstance(out_features, int) else tuple(out_features)
    if not isinstance(in_features, tuple):
        in_features = tuple(in_features) if isinstance(in_features, list) else (in_features,)
    if not isinstance(out_features, tuple):
        out_features = tuple(out_features) if isinstance(out_features, list) else (out_features,)

    weight_shape, features = shape(weight), (*in_features, *out_features)
    is_conv = tuple(weight_shape) != features
    if is_conv: kernel_shape = weight_shape[:-len(features)]

    if not is_conv:
        assert np.prod(in_features) % block_size == 0
    else:
        k = np.prod(kernel_shape)
        if k == 1:
            assert np.prod(in_features) % block_size == 0
        else:
            assert k % block_size == 0

    if is_conv and k != 1:
        mask = tf.greater(tf.random.uniform(shape=features, maxval=1, dtype=weight.dtype), p)
    else:
        mask = tf.greater(tf.random.uniform(shape=[np.prod(in_features) // block_size, *out_features], maxval=1, dtype=weight.dtype), p)
        mask = tf.reshape(tf.tile(mask[:, None], [1, block_size, * (1,) * len(out_features)]), features)

    if is_conv:
        mask = tf.tile(mask[(None,) * (len(weight_shape) - len(features))], (* kernel_shape, * (1,) * len(features)))

    weight = tf.where(mask, weight, tf.zeros_like(weight)) / (1 - p)

    return weight



