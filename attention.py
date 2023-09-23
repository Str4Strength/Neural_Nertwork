import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *
from .rot_pos_enc import *
from .linear import *



# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def link_memory(tensor, mask=None, tensor_saved=None, mask_saved=None, distance=None):
    if exists(tensor_saved):
        tensor = tf.concat([tensor_saved, tensor], axis=1)
        if exists(mask) and not exists(mask_saved): mask = tf.concat([tf.ones_like(tensor_saved, dtype=mask.dtype), mask], axis=1)
        if not exists(mask) and exists(mask_saved): mask = tf.concat([mask_saved, tf.ones_like(tensor, dtype=mask_saved.dtype)], axis=1)
        if exists(mask):
            sorting = tf.argsort(mask, axis=1)
            tensor, mask = tf.gather(tensor, sorting, batch_dims=1), tf.gather(mask, sorting, batch_dims=1)
        if isinstance(distance, int):
            cutting = tf.maximum(0, shape(tensor)[1] - distance)
            tensor = tensor[cutting:]
            if exists(mask): mask = mask[cutting:]
    return tensor, mask



def general_attention(q, k, v, mask=None, attn_map_bias=None):
    q_dot_k = tf.einsum('...qhd,...khd->...qkh', q, k) * tf.rsqrt(tf.cast(shape(q)[-1], q.dtype))
    if exists(attn_map_bias): q_dot_k += attn_map_bias
    if exists(mask): q_dot_k += -1e9 * (1.0 - mask)
    attn_weight = tf.nn.softmax(q_dot_k, axis=-2)
    tensor = tf.einsum('...qkh,...khd->...qhd', attn_weight, v)
    return tensor, attn_weight



def attention(
        query,
        key,
        value,
        out_features=None,
        hidden_features=None,
        heads=4,
        mask_query=None,
        mask_key=None,
        mask_value=None,
        saved_state=None,
        position_embedding=rotary_embedding,
        attention_bias=None,
        projection_bias=True,
        dropout=0.0,
        quantization=0.0,
        quantization_blocks=8,
        lrmul=1.0,
        component_projection=True,
        context_projection=True,
        query_weight_function=None,
        query_bias_function=None,
        key_weight_function=None,
        key_bias_function=None,
        value_weight_function=None,
        value_bias_function=None,
        out_weight_function=None,
        out_bias_function=None,
        trainable=True,
        scope='attention'
        ):
    with tf.variable_scope(scope):
        q_shape, k_shape, v_shape = shape(query), shape(key), shape(value)
        dtype = query.dtype
        if out_features is None: out_features = q_shape[-1]
        if hidden_features is None: hidden_features = out_features
        hidden_depth = hidden_features // heads


        # linear layers for query, key, value : B, S, C --> B, S, H, D
        if component_projection:
            q, q_m = linear(
                    query, out_features=[heads, hidden_depth], mask=mask_query, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=query_weight_function, bias_function=query_bias_function,
                    trainable=trainable, scope='projection_query')

            k, k_m = linear(
                    key, out_features=[heads, hidden_depth], mask=mask_key, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=key_weight_function, bias_function=key_bias_function,
                    trainable=trainable, scope='projection_key')

            v, v_m = linear(
                    value, out_features=[heads, hidden_depth], mask=mask_value, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=value_weight_function, bias_function=value_bias_function,
                    trainable=trainable, scope='projection_value')
        else:
            q = tf.reshape(query , q_shape[:-1] + [heads, hidden_features])
            q_m = tf.reshape(mask_query, q_shape[:-1] + [heads, hidden_features]) if exists(mask_query) else None

            k = tf.reshape(key, k_shape[:-1] + [heads, hidden_features])
            k_m = tf.reshape(mask_key, k_shape[:-1] + [heads, hidden_features]) if exists(mask_key) else None

            v = tf.reshape(value, v_shape[:-1] + [heads, hidden_features])
            v_m = tf.reshape(mask_value, v_shape[:-1] + [heads, hidden_features]) if exists(mask_value) else None

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
            setattr(saved_state, 'value_mask', v_m)

        # position embedding
        if exists(position_embedding):
            q = position_embedding(q, q_shape[-2], hidden_depth, trainable=trainable, scope='pos_emb_q')
            k = position_embedding(k, k_shape[-2], hidden_depth, trainable=trainable, scope='pos_emb_k')

        # masked softmax
        mask_attention = tf.einsum('bqh,bkh->bqkh',
                tf.reduce_max(q_m, axis=-1) if exists(mask_query) else tf.ones(shape(q)[:-1], dtype=dtype),
                tf.reduce_max(k_m, axis=-1) if exists(mask_key) else tf.ones(shape(k)[:-1], dtype=dtype))
        tensor, attn_weight = general_attention(q, k, v, mask=mask_attention, attn_map_bias=attention_bias)

        if context_projection:
            tensor, mask_tensor = linear(
                    tensor, in_features=shape(tensor)[-2:], out_features=out_features,
                    mask=q_m, bias=projection_bias, lrmul=lrmul, quantization=quantization,
                    quantization_blocks=hidden_features * quantization_blocks // q_shape[-1],
                    weight_function=out_weight_function, bias_function=out_bias_function,
                    trainable=trainable, scope='projection_out')
        else:
            tensor, mask_tensor = map(lambda t: tf.reshape(t, shape(t)[:-2] + [shape(t)[-2] * shape(t)[-1]]) if exists(t) else None, (tensor, q_m))

        return tensor, mask_tensor, attn_weight



