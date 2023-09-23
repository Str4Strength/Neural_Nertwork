import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *
from .rot_pos_enc import *
from .linear import *
from .normalization import *
from .attention import link_memory

from ..utils import tf_print


def bi_moment_attention(q, k, v, mask=None, attn_map_bias=None):
    q_dot_k = tf.einsum('...qhd,...khd->...qkh', q, k) * tf.rsqrt(tf.cast(shape(q)[-1], q.dtype))
    if exists(attn_map_bias): q_dot_k += attn_map_bias
    if exists(mask): q_dot_k += -1e9 * (1.0 - mask)
    attn_weight = tf.nn.softmax(q_dot_k, axis=-2)
    attn = tf.einsum('...qkh,...khd->...qhd', attn_weight, v)
    attn_square = tf.einsum('...qkh,...khd->...qhd', attn_weight, tf.square(v))
    return attn, attn_square


def adaptive_attention(
        tensor,
        reference,
        out_features=None,
        hidden_features=None,
        heads=4,
        mask_tensor=None,
        mask_reference=None,
        saved_state=None,
        position_embedding=rotary_embedding,
        attention_bias=None,
        projection_bias=True,
        quantization=0.0,
        quantization_blocks=8,
        lrmul=1.0,
        component_projection=True,
        context_projection=False,
        query_weight_function=None,
        query_bias_function=None,
        key_weight_function=None,
        key_bias_function=None,
        value_weight_function=None,
        value_bias_function=None,
        moment1_weight_function=None,
        moment1_bias_function=None,
        moment2_weight_function=None,
        moment2_bias_function=None,
        out_weight_function=None,
        out_bias_function=None,
        trainable=True,
        scope='adapttention',
        **norm_kwargs
        ):
    with tf.variable_scope(scope):
        t_shape, r_shape = shape(tensor), shape(reference)
        dtype = tensor.dtype
        if out_features is None: out_features = t_shape[-1]
        if hidden_features is None: hidden_features = out_features
        hidden_depth = hidden_features // heads

        if shape(tensor)[-1] != out_features:
            tensor = linear(
                    tensor, out_features=out_features, mask=mask_tensor, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=t_shape[-1],
                    weight_function=out_weight_function, bias_function=out_bias_function,
                    trainable=trainable, scope='projection_in')

        tensor = normalization(tensor, mask=mask_tensor, trainable=trainable, scope='norm_content', **norm_kwargs)
        ref_norm = normalization(reference, mask=mask_reference, trainable=trainable, scope='norm_style', **norm_kwargs)

        if component_projection:
            q, q_m = linear(
                    tensor, out_features=[heads, hidden_depth], mask=mask_tensor, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=query_weight_function, bias_function=query_bias_function, trainable=trainable,
                    scope='projection_query')

            k, k_m = linear(
                    ref_norm, out_features=[heads, hidden_depth], mask=mask_reference, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=key_weight_function, bias_function=key_bias_function, trainable=trainable,
                    scope='projection_key')

            v, v_m = linear(
                    reference, out_features=[heads, hidden_depth], mask=mask_reference, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=value_weight_function, bias_function=value_bias_function, trainable=trainable,
                    scope='projection_value')

        else:
            q = tf.reshape(query , t_shape[:-1] + [heads, hidden_features])
            q_m = tf.reshape(mask_query, t_shape[:-1] + [heads, hidden_features]) if exists(mask_query) else None

            k = tf.reshape(key, r_shape[:-1] + [heads, hidden_features])
            k_m = tf.reshape(mask_key, r_shape[:-1] + [heads, hidden_features]) if exists(mask_key) else None

            v = tf.reshape(value, r_shape[:-1] + [heads, hidden_features])
            v_m = tf.reshape(mask_value, r_shape[:-1] + [heads, hidden_features]) if exists(mask_value) else None

        if exists(saved_state):
            if exists(getattr(saved_state, 'key', None)): input_relation = 'cross'
            distance = getattr(saved_state, 'distance', None)

            k_saved, k_m_saved = getattr(saved_state, 'key', None), getattr(saved_state, 'key_mask', None)
            k, k_m = link_memory(k, k_m, tensor_saved=k_saved, mask_saved=k_m_saved, distance=distance)
            setattr(saved_state, 'key', k)
            setattr(saved_state, 'key_mask', k_m)

            v_saved, v_m_saved = getattr(saved_state, 'value', None), getattr(saved_state, 'value_mask', None)
            v, v_m = link_memory(v, v_m, tensor_saved=v_saved, mask_saved=v_m_saved, distance=distance)
            setattr(saved_state, 'value', v)
            setattr(saved_state, 'valuie_mask', v_m)

        # position embedding
        if exists(position_embedding):
            q = position_embedding(q, t_shape[-2], hidden_depth, trainable=trainable, scope='pos_emb_q')
            k = position_embedding(k, r_shape[-2], hidden_depth, trainable=trainable, scope='pos_emb_k')

        # masked softmax
        mask_attention = tf.einsum('bqh,bkh->bqkh',
                tf.reduce_max(q_m, axis=-1) if exists(mask_tensor) else tf.ones(shape(q)[:-1], dtype=dtype),
                tf.reduce_max(k_m, axis=-1) if exists(mask_reference) else tf.ones(shape(k)[:-1], dtype=dtype))
        moment_1st, square_1st = bi_moment_attention(q, k, v, mask=mask_attention, attn_map_bias=attention_bias)

        moment_2nd = tf.concat([tf.square(moment_1st), square_1st], axis=-1)
        mask_1st = reconstruct_mask(shape(v)[-1], q_m, axis=-1) if exists(mask_tensor) else None
        mask_2nd = reconstruct_mask(2 * shape(v)[-1], q_m, axis=-1) if exists(mask_tensor) else None

        adaptive_1st, mask_1st = linear(
                moment_1st, in_features=[heads, shape(v)[-1]], out_features=out_features, mask=mask_1st, bias=projection_bias,
                lrmul=lrmul, quantization=quantization, quantization_blocks=shape(v)[-1],
                weight_function=moment1_weight_function, bias_function=moment1_bias_function, trainable=trainable,
                scope='projection_1st_moment')

        adaptive_2nd, mask_2nd = linear(
                moment_2nd, in_features=[heads, shape(v)[-1] * 2], out_features=out_features, mask=mask_2nd, bias=projection_bias,
                lrmul=lrmul, quantization=quantization, quantization_blocks=shape(v)[-1] * 2,
                weight_function=moment2_weight_function, bias_function=moment2_bias_function, trainable=trainable,
                scope='prjection_2nd_moment')

        def _mean_(x, mask=None, **kwargs):
            if mask is None: return tf.reduce_mean(x, **kwargs)
            mask_sum = tf.reduce_sum(mask, **kwargs)
            return tf.reduce_sum(x, **kwargs) / tf.where(mask_sum > 0, mask_sum, tf.ones_like(mask_sum))

        adaptive_1st, adaptive_2nd = _mean_(adaptive_1st, mask=mask_1st), _mean_(adaptive_2nd, mask=mask_2nd)

        tensor *= tf.sqrt(tf.nn.softplus(adaptive_2nd) + 1e-8)
        tensor += adaptive_1st

        mask_tensor = reconstruct_mask(out_features, mask_tensor, axis=-1)
        if exists(mask_tensor): tensor *= mask_tensor

        if context_projection:
            tensor, mask_tensor = linear(
                    tensor, out_features=out_features, mask=mask_tensor, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=out_features * quantization_blocks // t_shape[-1],
                    weight_function=out_weight_function, bias_function=out_bias_function,
                    trainable=trainable, scope='projection_out')

        return tensor, mask_tensor

