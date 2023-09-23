import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *
from .variable_setting import *

from ..utils import tf_print

# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def normalization(
        tensor,
        groups=None,
        group_size=None,
        batch=False,
        scale=True,
        shift=True,
        gamma=None,
        beta=None,
        mask=None,
        epsilon=1e-5,
        momentum=0.1,
        lrmul=1.0,
        gamma_function=None,
        beta_function=None,
        trainable=True,
        scope='normalization'
        ):
    with tf.variable_scope(scope):
        dtype, tensor_shape = tensor.dtype, shape(tensor)
        len_shape = len(tensor_shape)
        if exists(groups) and exists(group_size):
            assert tensor_shape[-1] == groups * group_size
        else:
            assert exists(groups) or exists(group_size)
            groups = tensor_shape[-1] // group_size if exists(group_size) else groups
            group_size = tensor_shape[-1] // groups if exists(groups) else group_size

        var_shape = (1,) * (len_shape - 1) + (groups, 1)

        if scale and gamma is None: gamma = weight_(var_shape, dtype, init=tf.ones_initializer, lrmul=lrmul,
                    function=gamma_function, trainable=trainable, scope='gamma')
        if shift and beta is None: beta = bias_(var_shape, dtype, lrmul=lrmul,
                function=beta_function, trainable=trainable, scope='beta')

        # ... d g
        reduce_kwargs = {'axis': (0,) * int(batch) + tuple(range(1, len_shape - 1)) + (len_shape,), 'keepdims': True}

        def _mean_(x, mask=None, **kwargs):
            if mask is None: return tf.reduce_mean(x, **kwargs)
            mask_sum = tf.reduce_sum(mask, **kwargs)
            return tf.reduce_sum(x, **kwargs) / tf.where(mask_sum > 0, mask_sum, tf.ones_like(mask_sum))

        grouped_tensor = tf.reshape(tensor, tensor_shape[:-1] + [groups, group_size])
        grouped_mask = tf.reshape(mask, tensor_shape[:-1] + [groups, group_size]) if exists(mask) else None

        smp_mean = _mean_(grouped_tensor, mask=grouped_mask, **reduce_kwargs)     # 1, ..., 1, g, 1
        smp_variance = _mean_(grouped_tensor ** 2.0, mask=grouped_mask, **reduce_kwargs) - (smp_mean ** 2.0)

        if batch:
            ema_mean = tf.get_variable('ema_mean', var_shape, dtype, initializer=tf.zeros_initializer,
                    trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])
            ema_variance = tf.get_variable('ema_variance', var_shape, dtype, initializer=tf.ones_initializer,
                    trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])

            if exists(mask): reduced_mask = tf.reduce_max(grouped_mask, **reduce_kwargs)[None, None]
            mean, variance = momentum * (smp_mean - ema_mean), momentum * (smp_variance - ema_variance)
            if exists(mask): mean, variacne = map(lambda t: t * reduced_mask, (mean, variance))
            mean, variance = mean + ema_mean, variance + ema_variance

            with tf.control_dependencies([tf.assign(ema_mean, mean), tf.assign(ema_variance, variance)]
                    if trainable else []):
                smp_mean, smp_variance = tf.identity(mean), tf.identity(variance)

        def _duplicate_(x, mask=None,):
            x = tf.reshape(tf.tile(x, (1,) * len_shape + (group_size,)), (* shape(x)[:-2], tensor_shape[-1]))
            if exists(mask): x *= mask
            return x

        tensor = (tensor - _duplicate_(smp_mean, mask=mask)) * tf.rsqrt(_duplicate_(tf.maximum(smp_variance, epsilon)))
        if scale: tensor *= _duplicate_(gamma, mask=mask)
        if shift: tensor += _duplicate_(beta, mask=mask)
        if exists(mask): tensor *= mask

    return tensor



