import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import gen_nn_ops

from .function import *
from .variable_setting import *


# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def convolution(
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
                             'Received groups={}, but the input has {} channels (full input shape is {}).'.format(
                groups, tensor_shape[-1], tensor_shape))

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

        weight = weight_(kernels + [tensor_shape[-1]//groups, filters//groups], dtype, lrmul=lrmul,
                function=weight_function, trainable=trainable)
        if trainable: weight = quantization_noise(weight, tensor_shape[-1]//groups, filters, quantization, quantization_blocks)
        if groups > 1: weight = tf.tile(weight, [1] * rank + [1, groups])

        # manufacture shape
        tensor = tf.reshape(tensor, [-1] + tensor_shape[n_batch_dims:])
        if exists(mask): mask = tf.reshape(mask, [-1] + tensor_shape[n_batch_dims:])

        if data_format == 'channels_first': tensor = tf.transpose(tensor, [0] + list(range(2, rank + 2)) + [1])

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

        if exists(mask):
            mask = tf.nn.pool(mask, kernels, 'MAX', strides=strides[1:-1], padding=padding, data_format=data_format,
                    dilations=dilations[1:-1] if padding != 'SAME' else None)

        # selection
        ops = gen_nn_ops.conv3d if rank == 3 else gen_nn_ops.conv2d

        if rank == 1:
            tensor = tf.expand_dims(tensor, axis=1)
            weight = weight[None, Ellipsis]
            strides = [strides[0], 1] + strides[1:]
            dilations = [dilations[0], 1] + dilations[1:]

        # perform operation
        tensor = ops(tensor, weight, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC', dilations=dilations)
        if rank == 1: tensor = tf.squeeze(tensor, axis=[1])

        # bias
        if bias: tensor += bias_(([1]*(rank + 1) + [filters]), dtype, lrmul=lrmul, function=bias_function, trainable=trainable)

        # recover shape
        recover_shape = shape(tensor)

        if data_format == 'channels_first':
            tensor = tf.transpose(tensor, [0, rank + 1] + list(range(1, rank + 1)))
            batch_extend = tensor_shape[:n_batch_dims] + [filters] + recover_shape[1:-1]
        else:
            batch_extend = tensor_shape[:n_batch_dims] + recover_shape[1:]

        mask = reconstruct_mask(filters, mask=mask, axis=1 if data_format == 'channels_first' else -1)

        tensor = tf.reshape(tensor, batch_extend)
        if exists(mask):
            mask = tf.reshape(mask, batch_extend)
            tensor *= mask

    return tensor, mask



