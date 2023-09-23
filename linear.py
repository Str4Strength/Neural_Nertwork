import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *
from .variable_setting import *


# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def linear(
        tensor,
        in_features=None,
        out_features=None,
        mask=None,
        bias=True,
        lrmul=1.0,
        quantization=0.0,
        quantization_blocks=8,
        weight_function=None,
        bias_function=None,
        trainable=True,
        scope='linear'
        ):
    # mask must be in a full-shape except channels
    with tf.variable_scope(scope):
        dtype, tensor_shape = tensor.dtype, shape(tensor)

        if in_features is None: in_features = (tensor_shape[-1], )
        in_features = (in_features,) if isinstance(in_features, int) else tuple(in_features)
        if out_features is None: out_features = in_features
        out_features = (out_features,) if isinstance(out_features, int) else tuple(out_features)

        dims, ins, outs = len(tensor_shape), len(in_features), len(out_features)

        weight = weight_((*in_features, *out_features), dtype, lrmul=lrmul, function=weight_function, trainable=trainable)
        if trainable: weight = quantization_noise(weight, in_features, out_features, quantization, quantization_blocks)

        axes = (list(range(dims - ins, dims)), list(range(ins)))
        tensor = tf.tensordot(tensor, weight, axes)

        if bias: tensor += bias_(out_features, dtype, lrmul=lrmul, function=bias_function, trainable=trainable)[(None,) * (dims - ins)]

        if exists(mask):
            mask = tf.reduce_max(mask, axis=tuple(range(dims - ins, dims)))
            mask = tf.tile(mask[(Ellipsis, * (None,) * outs)], (* (1,) * (dims - ins), *out_features))
            tensor *= mask

        return tensor, mask



