import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import gen_nn_ops

import numpy as np

from .function import *
from .variable_setting import *


# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def deconvolution(
        tensor,
        rank,
        filters,
        kernels,
        strides=1,
        dilations=1,
        padding='SAME',
        groups=1,
        mask=None,
        bias=True,
        lrmul=1.0,
        quantization=0.0,
        quantization_blocks=8,
        weight_function=None,
        bias_function=None,
        data_format=None,
        trainable=True,
        scope='convolution'
        ):
    """
    convolution of <rank>-dimensional
    """
    with tf.variable_scope(scope):
        assert len(shape(tensor)) > rank + 1

        # padding check
        padding = padding.upper()

        # rank check
        if rank not in {1, 2, 3}:
            raise ValueError('The number of spatial dimensions must be one of 1, 2 or 3 but saw {}.'.format(rank))

        # filters check
        if isinstance(filters, float): filters = int(filters)
        if exists(filters) and filters % groups != 0:
            raise ValueError('The number of filters must be evenly divisible by the number of groups.'
                             'Received: groups={}, filters={}.'.format(groups, filters))

        # channels check
        dtype = tensor.dtype
        tensor_shape = shape(tensor)
        if tensor_shape[-1] % groups != 0:
            raise ValueError('The number of input channels must be evenly divisible by the number of groups.'
                             'Received groups={}, but the input has {} channels (full input shape is {}).'.format(groups, tensor_shape[-1], tensor_shape))

        # kernel size control
        if isinstance(kernels, int): kernels = [kernels, ] * rank
        kernels = list(kernels)
        if len(kernels) != rank:
            raise ValueError('The `kernels` argument must be a list of {} integers.'
                             'Received: {}.'.format(rank, kernels))
        for single_size in kernels:
            assert isinstance(single_size, int)
        if not all(kernels):
            raise ValueError('`kernels` cannot contain 0(s).'
                             'Received: {}'.format(kernels))

        # internal convolution operation
        n_total_dims = len(tensor_shape)
        n_batch_dims = n_total_dims - rank - 1
        batch_dims = list(range(0, n_batch_dims))

        # mask
        if exists(mask): tensor = tensor * mask

        weight = weight_(kernels + [filters//groups, tensor_shape[-1]//groups], dtype, lrmul=lrmul,
                function=weight_function, trainable=trainable)
        if trainable: weight = quantization_noise(weight, filters//groups, tensor_shape[-1], quantization, quantization_blocks)
        if groups > 1: weight = tf.tile(weight, [1] * rank + [1, groups])

        # manufacture shape
        tensor = tf.reshape(tensor, [-1] + tensor_shape[n_batch_dims:])
        if exists(mask): mask = tf.reshape(mask, [-1] + tensor_shape[n_batch_dims:])

        if data_format == 'channels_first':
            tensor = tf.transpose(tensor, [0] + list(range(2, rank + 2)) + [1])
            if exists(mask): mask = tf.transpose(mask, [0] + list(range(2, rank + 2)) + [1])

        def reform(values, name='values'):
            if isinstance(values, int): values = [values, ] * rank
            values = list(values)

            for single_size in values: assert isinstance(single_size, int)

            if not all(values): raise ValueError('`{}` cannot contain 0(s). Received: {}'.format(name, values))

            n_value_dims = len(values)

            if n_value_dims != (rank + 2):
                if n_value_dims == 1:
                    values = values * rank
                elif n_value_dims != rank:
                    raise ValueError("{} must be length 1, {} or {} but was {}.".format(name, rank, n_total_dims,
                        n_value_dims))

                values = [1] + values + [1]

            return values

        # strides
        strides = [1] * (rank + 2) if strides is None else reform(strides, 'strides')

        # dilations
        dilations = [1] * (rank + 2) if dilations is None else reform(dilations, 'dilations')

        # calculating output shape
        output_shape = shape(tensor)[:1]

        valid_spatial = lambda l, k, s, d: (l - 1) * s + k + (k-1) * (d-1)
        if padding == 'VALID':
            spatials = list(map(valid_spatial, shape(tensor)[1:-1], kernels, strides[1:-1], dilations[1:-1]))
        else:
            spatials = list(map(lambda l, s: l * s, shape(tensor)[1:-1], strides[1:-1]))

        output_shape += spatials + [filters]

        if exists(mask):
            mask_shape = shape(mask)
            f_, *mask_spatials, b_ = mask_shape
            mask_strides = strides.copy()
            for n in range(rank): mask_strides.insert(2 * n + 1, 1)
            for n in range(rank): mask_spatials.insert(2 * n + 1, 1)
            mask = tf.reshape(mask, [f_, *mask_spatials, b_])

            if padding == 'SAME':
                mask = tf.reshape(tf.tile(mask, mask_strides), list(map(lambda a, b: a * b, mask_shape, strides)))
            else:
                mask = tf.slice(mask, [0] * (len(mask_spatials) + 2), list(map(lambda s: tf.maximum(s - 1, 1), [f_, *mask_spatials, b_])))
                mask = tf.reshape(tf.tile(mask, mask_strides), list(map(lambda a, b: (a - 1) * b, mask_shape, strides)))
                pad_values = list(zip((np.asarray(kernels) - 1) * (np.asarray(dilations[1:-1]) - 1), np.asarray(kernels)))
                mask = tf.pad(mask, ((0, 0), *pad_values, (0, 0)))

        # selection and perform operation
        ops = gen_nn_ops.conv3d_backprop_input_v2 if rank == 3 else gen_nn_ops.conv2d_backprop_input

        if rank == 1:
            tensor = tf.expand_dims(tensor, axis=1)
            weight = weight[None, Ellipsis]
            strides = [strides[0], 1] + strides[1:]
            dilations = [dilations[0], 1] + dilations[1:]
            output_shape = [output_shape[0], 1] + output_shape[1:]

        tensor = ops(output_shape, weight, tensor, strides, padding, use_cudnn_on_gpu=True, dilations=dilations)
        if rank == 1: tensor = tf.squeeze(tensor, axis=[1])

        # bias
        if bias: tensor += bias_(([1]*(rank + 1) + [filters]), dtype, lrmul=lrmul, function=bias_function, trainable=trainable)

        # recover shape
        recover_shape = shape(tensor)

        if data_format == 'channels_first':
            tensor = tf.transpose(tensor, [0, rank + 1] + list(range(1, rank + 1)))
            if exists(mask): mask = tf.transpose(mask, [0, rank + 1] + list(range(1, rank + 1)))
            batch_extend = tensor_shape[:n_batch_dims] + [filters] + recover_shape[1:-1]
        else:
            batch_extend = tensor_shape[:n_batch_dims] + recover_shape[1:]

        mask = reconstruct_mask(filters, mask=mask, axis=1 if data_format == 'channels_first' else -1)

        tensor = tf.reshape(tensor, batch_extend)
        if exists(mask):
            mask = tf.reshape(mask, batch_extend)
            tensor *= mask

    return tensor, mask



