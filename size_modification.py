import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *



# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def upsampolate(
        tensor,
        features,
        axis=1,
        up_rate=2,
        mask=None,
        ):
    # strictrly required form of, single batch size B frontmost, and single channels C backmost
    t_shape = shape(tensor)
    max_axis = len(t_shape) - 1
    if axis < 0: axis += len(t_shape)
    assert 0 < axis and axis < max_axis

    tensor_tile = tf.tile(tf.expand_dims(tensor, axis + 1), [1] * (axis + 1) + [up_rate] + [1] * (max_axis - axis))
    tensor_expand = tf.reshape(tensor_tile, t_shape[:axis] + [up_rate * t_shape[axis]] + t_shape[axis + 1:])
    kernel_size = 2 * up_rate
    pad_left, pad_right = (kernel_size - 1) // 2, kernel_size // 2
    tensor_pad = tf.concat([tf.gather(tensor, [0] * pad_left, axis=axis, batch_dims=0), tensor_expand,
        tf.gather(tensor, [t_shape[axis] - 1] * pad_right, axis=axis, batch_dims=0)], axis=axis)
    filters = tf.ones([kernel_size, t_shape[-1], features], dtype=tensor.dtype) / (kernel_size * t_shape[-1])
    if max_axis == 2: # 1d
        upsampolated = tf.nn.conv1d(tensor_pad, filters, stride=1, padding='VALID')
    elif max_axis == 3: # 2d, axis=1 or axis=2
        upsampolated = tf.nn.conv2d(tensor_pad, tf.expand_dims(filters, 2 - axis), strides=(1, 1), padding='VALID')
    else:
        raise ValueError("1d and 2d only implemented")

    if exists(mask):
        mask_tile = tf.tile(tf.expand_dims(mask, axis + 1), [1] * (axis + 1) + [up_rate] + [1] * (max_axis - axis))
        mask_expand = tf.reshape(mask_tile, t_shape[:axis] + [up_rate * t_shape[axis]] + t_shape[axis + 1:])
        mask = reconstruct_mask(features, mask_expand, axis=-1)
        upsampolated *= mask

    return upsampolated, mask



