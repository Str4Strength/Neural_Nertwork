import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import random_ops as tfrop

import numpy as np

import scipy.signal

import re
import math
import random
import string

from functools import partial
from termcolor import cprint

from .function import *
from .linear import *
from .rot_pos_enc import *



# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def shift(token, amount, mask=None):
    if amount == 0: return token
    if exists(mask): token *= tf.broadcast_to(mask, shape(token))
    token_len = shape(token)[1]
    token = tf.pad(token, [[0, 0], [max(amount, 0), max(-amount, 0)]] + [[0, 0]] * (len(shape(token)) - 2))
    return token[:, min(0, -amount):min(0, -amount) + token_len,]


def pre_shift_token(
        token,
        shifts,
        function,
        **kwargs):
    mask = kwargs.get('mask', None)
    segments = len(shifts)
    features_per_shift = shape(token)[-1] // segments
    splitted = tf.split(token, features_per_shift, axis=-1)
    segments_to_shift, rest = splitted[:segments], splitted[segments:]
    segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
    token = tf.conat((*segments_to_shift, *rest), axis=-1)
    return function(token, **kwargs)


def process_checker(q_procs, k_procs, v_procs):
    lenproc_q = len(q_procs) if exists(q_procs) else 0
    lenproc_k = len(k_procs) if exists(k_procs) else 0
    lenproc_v = len(v_procs) if exists(v_procs) else 0
    lenproc_qk, lenproc_kv, lenproc_vq, lenproc_qkv = lenproc_q * lenproc_k, lenproc_k * lenproc_v, lenproc_v * lenproc_q, lenproc_q * lenproc_k * lenproc_v

    if lenproc_qkv != 0:
        assert (lenproc_q == lenproc_k) and (lenproc_k == lenproc_v)
    else:
        if lenproc_qk != 0:
            assert lenproc_q == lenproc_k
            v_procs = [None] * lenproc_k
        elif lenproc_kv != 0:
            assert lenproc_k == lenproc_v
            q_procs = [None] * lenproc_v
        elif lenproc_vq != 0:
            assert lenproc_v == lenproc_q
            k_procs = [None] * lenproc_q
        elif lenproc_q != 0:
            k_procs, v_procs = [[None] * lenproc_q] * 2
        elif lenproc_k != 0:
            v_procs, q_procs = [[None] * lenproc_k] * 2
        elif lenproc_v != 0:
            q_procs, k_procs = [[None] * lenproc_v] * 2
        else:
            q_procs, k_procs, v_procs = [[None]] * 3

    return q_procs, k_procs, v_procs


def apply_process(t, t_procs, mask=None):
    if exists(mask):
        processed = [f(t, mask=mask) if exists(f) else t for f in t_procs]
    else:
        processed = [f(t) if exists(f) else t for f in t_procs]
    return processed


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


# kernel functions
def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, epsilon=1e-4):
    b, i, h, d = shape(data)

    data_normalizer = tf.rsqrt(tf.sqrt(tf.cast(d, data.dtype))) if normalize_data else 1.

    ratio = tf.rsqrt(tf.cast(shape(projection_matrix)[0], data.dtype))

    #data_dash = tf.einsum('...ihd,jd->...ihj', (data_normalizer * data), projection_matrix)
    data_dash = tf.tensordot(data_normalizer * data, projection_matrix, [-1, -1])

    diag_data = data ** 2
    diag_data = tf.reduce_sum(diag_data, axis=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data[Ellipsis, None]

    reduce_axis = -1 if is_query else [-3, -1]
    data_dash = tf.exp(data_dash - diag_data - tf.reduce_max(data_dash, axis=reduce_axis, keepdims=True) + epsilon)
    data_dash *= ratio

    return data_dash


def generalized_kernel(data, *, projection_matrix, kernel_function = tf.nn.relu, normalize_data=True, epsilon=1e-3):
    b, i, h, d = shape(data)

    data_normalizer = tf.rsqrt(tf.sqrt(tf.cast(d, data.dtype))) if normalize_data else 1.

    if not exists(projection_matrix): return kernel_function(data_normalizer * data) + epsilon

    #data_dash = tf.einsum('...ihd,jd->...ihj', (data_normalizer * data), projection_matrix)
    data_dash = tf.tensordot(data_normalizer * data, projection_matrix, [-1, -1])

    data_prime = kernel_function(data_dash) + epsilon

    return data_prime


def orthogonal_matrix_chunk(columns, dtype):
    unstructured_block = tf.random.normal((columns, columns), dtype=dtype)

    q, r = tf.linalg.qr(unstructured_block)

    return tf.transpose(q, [1, 0])


def rotational_products_chunk(columns, dtype):
    rotations = columns * np.ceil(np.log(columns))

    q = np.eye(columns, columns)

    for _ in range(rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(columns, 2)
        index_i, index_j = min(random_indices[0], random_indices[1]), max(random_indices[0], random_indices[1])
        slice_i, slice_j = q[index_i], q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.sin(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_j + math.cos(random_angle) * slice_j
        q[index_i], q[index_j] = new_slice_i, new_slice_j

    return tf.constant(q, dtype=dtype)


def gaussian_orthogonal_random_matrix(rows, columns, dtype=tf.float64, scaling=0, struct_mode=False):
    full_blocks = int(rows / columns)
    block_list = []

    create_function = rotational_products_chunk if struct_mode else orthogonal_matrix_chunk

    for _ in range(full_blocks):
        q = create_function(columns, dtype)
        block_list.append(q)

    remaining_rows = rows - full_blocks * columns
    if remaining_rows > 0:
        q = create_function(columns, dtype)
        block_list.append(q[:remaining_rows])

    final_matrix = tf.concat(block_list, axis=0)

    if scaling == 0:
        multiplier = tf.norm(tf.random.normal((rows, columns)), axis=1)
    elif scaling == 1:
        multiplier = tf.math.sqrt(float(columns) * tf.ones(rows, dtype=dtype))
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    #return tf.einsum('rr,rc->rc', tf.linalg.diag(multiplier), final_matrix)
    return tf.tensordot(tf.linalg.diag(multiplier), final_matrix, [1, 0])


def noncausal_linear_attention(q, k, v, mask=None):
    #k_sum = tf.einsum("...lhd->...hd", k)
    #denominator = 1.0 / tf.einsum("...lhd,...hd->...lh", q, k_sum)
    k_sum = tf.reduce_sum(k, axis=-3, keepdims=True)
    denominator = 1.0 / tf.reduce_sum(q * k_sum, axis=-1, keepdims=True)
    if exists(mask):
        denom_zeros = tf.equal(tf.reduce_max(mask, axis=-1, keepdims=True), 0)
        denominator = tf.where(denom_zeros, tf.zeros_like(denominator), denominator)

    #context = tf.einsum("...lhk,...lhv->...hkv", k, v)
    #attention = tf.einsum("...hdo,...lhd,...lh->...lho", context, q, denominator)
    context = tf.reduce_sum(k[Ellipsis, None] * v[Ellipsis, None, :], axis=-4, keepdims=True)
    attention = tf.reduce_sum((q * denominator)[Ellipsis, None] * context, axis=-2)

    return attention


#@tf.custom_gradient     #TODO update gradient for next refer to layers.py, if needed
def causal_linear_attention(q, k, v, mask=None, gate=None, chunks=128, epsilon=1e-6):
    last_k_sum = 0
    last_context_sum = 0
    attention = []

    if exists(gate):
        gate_chunk = chunk_(gate, chunks, axis=-2)
        if exists(mask): mask_chunk = chunk_(mask, chunks, axis=-3)
        for i, q, k, v in enumerate(zip(*map(lambda t: chunk_(t, chunks, axis=-3), (q, k, v)))):
            g = gate_chunk[i], r_g = 1.0 - gate_chunk[i]
            g_cat = tf.concat([tf.ones_like(g[Ellipsis, :1, :], g.dtype), g], axis=-2)
            rg_cat = tf.concat([tf.ones_like(r_g[Ellipsis, :1, :], r_g.dtype), r_g], axis=-2)

            g_rev_cump = tf.cumprod(g_cat, axis=-2, exclusive=True, reverse=True)

            k_cat = tf.concat([last_k_sum, k], axis=-3)
            #k_sum = tf.cumsum(tf.einsum("...ch,...ch,...chd->...chd", rg_cat, g_rev_cump, k_cat), axis=-3)
            k_sum = tf.cumsum(rg_cat[Ellipsis, None] * g_rev_cump[Ellipsis, None] * k_cat, axis=-3)
            k_sum = k_sum[Ellipsis, 1:, :, :]
            #denominator = 1.0 / tf.einsum("...chd,...chd->...ch", q, k_sum + epsilon)
            denominator = 1.0 / tf.reduce_sum(q * (k_sum + epsilon), axis=-1, keepdims=True)
            if exists(mask):
                denom_zeros = tf.equal(tf.reduce_max(mask_chunk[i], axis=-1, keepdims=True), 0)
                denominator = tf.where(denom_zeros, tf.zeros_like(denominator), denominator)

            #context_cat = tf.concat([last_context_sum, tf.einsum("...chk,...chv->...chkv", k, v)], axis=-4)
            #context_sum = tf.cumsum(tf.einsum("...ch,...ch,...chdo->...chdo", rg_cat, g_rev_cump, context_cat), axis=-4)
            #context_sum = context_sum[Ellipsis, 1:, :, :, :]
            context_cat = tf.concat([last_context_sum, k[Ellipsis, None] * v[Ellipsis, None, :]], axis=-4)
            context_sum = tf.cumsum(rg_cat[Ellipsis, None, None] * g_rev_cump[Ellipsis, None, None], context_cat, axis=-4)
            context_sum = context_sum[Ellipsis, 1:, :, :, :]

            #attn = tf.einsum("...chdo,...chd,...ch,->...cho", context_sum, q, denominator)
            attn = tf.reduce_sum(context_sum * q[Ellipsis, None], axis=-2) * denominator

            last_k_sum = k_sum[Ellipsis, -1:, :, :]
            last_context_sum = context_sum[Ellipsis, -1:, :, :, :]
            attention.append(attn)

    else:
        for q, k, v in zip(*map(lambda t: chunk_(t, chunks, axis=-3), (q, k, v))):
            k_sum = last_k_sum + tf.cumsum(k, axis=-3)
            #denominator = 1.0 / tf.einsum("...chd,...chd->...ch", q, k_sum + epsilon)
            denominator = 1.0 / tf.reduce_sum(q * (k_sum + epsilon), axis=-1, keepdims=True)
            if exists(mask):
                denom_zeros = tf.equal(tf.reduce_max(mask, axis=-1, keepdims=True), 0)
                denominator = tf.where(denom_zeros, tf.zeros_like(denominator), denominator)

            #context = tf.einsum("...chk,...chv->...chkv", k, v)
            context = k[Ellipsis, None] * v[Ellipsis, None, :]
            context_sum = last_context_sum + tf.cumsum(context, axis=-4)

            #attn = tf.einsum("...chdo,...chd,...ch->...cho", context_sum, q, denominator)
            attn = tf.reduce_sum(context_sum * q[Ellipsis, None], axis=-2) * denominator

            last_k_sum = k_sum[Ellipsis, -1:, :, :]
            last_context_sum = context_sum[Ellipsis, -1:, :, :, :]
            attention.append(attn)

    return tf.concat(attention, axis=-3)


def fast_attention(
        query,
        key,
        value,
        out_features=None,
        hidden_features=None,
        heads=4,
        mask_query=None,
        mask_key=None,
        mask_value=None,
        causal=False,
        causal_chunks=128,
        gates=True,
        position_embedding=rotary_embedding,
        orthogonal_random_features=True,
        orthogonal_scaling=0,
        kernel_regularization=tf.nn.relu,
        saved_state=None,
        projection_bias=True,
        quantization=0.0,
        quantization_blocks=8,
        lrmul=1.0,
        query_processes=None,
        key_processes=None,
        value_processes=None,
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
        scope='fast_attn',
        ):
    with tf.variable_scope(scope):
        query_processes, key_processes, value_processes = process_checker(query_processes, key_processes, value_processes)

        q_shape, k_shape, v_shape, dtype = shape(query), shape(key), shape(value), query.dtype

        if not isinstance(heads, int): raise ValueError("The number of heads must be integer, but given {}".format(type(heads)))

        out_features = int(out_features) if exists(out_features) else q_shape[-1]
        if out_features % heads != 0: raise ValueError("The number of heads must divide out_units evenly, but heads:{} and out_units: {}".format(heads, out_features))

        hidden_features = int(hidden_features) if exists(hidden_features) else out_features
        if hidden_features % heads != 0: raise ValueError("The number of heads must divide hidden_units evenly, but heads:{} and hidden_units: {}".format(heads, hidden_features))

        hidden_depth = hidden_features // heads
        theta_random_features = hidden_depth * int(np.ceil(np.log(hidden_depth)))

        gates = gates and causal

        # material projection
        if component_projection:
            q, q_m = linear(
                    query, out_features=[heads, hidden_depth], mask=mask_query, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=query_weight_function, bias_function=query_bias_function, trainable=trainable,
                    scope='projection_query')

            k, k_m = linear(
                    key, out_features=[heads, hidden_depth], mask=mask_key, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=key_weight_function, bias_function=key_bias_function, trainable=trainable,
                    scope='projection_key')

            v, v_m = linear(
                    value, out_features=[heads, hidden_depth], mask=mask_value, bias=projection_bias,
                    lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                    weight_function=value_weight_function, bias_function=value_bias_function, trainable=trainable,
                    scope='projection_value')

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

        if gates:
            gate, gate_m = linear(
                    key, out_features=heads, mask=mask_key, bias=True, lrmul=lrmul, trainable=trainable, scope='projection_gate')
            gate = tf.math.sigmoid(gate)
            if exists(gate_m): gate *= gate_m
            if exists(saved_state):
                gate_saved, gate_m_saved = getattr(saved_state, 'gate', None), getattr(saved_state, 'gate_mask', None)
                gate, gate_m = link_memory(gate, gate_m, tensor_saved=gate_saved, mask_saved=gate_m_saved, distance=distance)
                setattr(saved_state, 'gate', gate)
                setattr(saved_state, 'gate_mask', gate_m)
        else:
            gate = None

        if orthogonal_random_features:
            saved_matrix = tf.get_variable("projection_matrix", shape=[theta_random_features, hidden_depth],
                    dtype=dtype, initializer=tf.initializers.random_normal(0, 1), trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])

            if trainable:
                projection_matrix = gaussian_orthogonal_random_matrix(theta_random_features, hidden_depth,
                        dtype=dtype, scaling=orthogonal_scaling, struct_mode=False)

                def update():
                    with tf.control_dependencies([tf.assign(saved_matrix, projection_matrix)]):
                        return tf.identity(projection_matrix)

                condition = tf.abs(tf.reduce_mean(projection_matrix)) > tf.abs(tf.reduce_mean(saved_matrix))
                projection_matrix = tf.cond(condition, update, lambda: tf.identity(projection_matrix))
            else:
                projection_matrix = saved_matrix

        processed_q = apply_process(q, query_processes, mask=q_m)
        processed_k = apply_process(k, key_processes, mask=k_m)
        processed_v = apply_process(v, value_processes, mask=v_m)

        tensors, masks_tensor = [], []
        for n, (q, k, v) in enumerate(list(zip(processed_q, processed_k, processed_v))):

            # orthogonal_random_features & kernel_regularization
            if not orthogonal_random_features:
                q = tf.softmax(q, axis=-1)
                k = tf.exp(k - tf.reduce_max(k, axis=-3, keepdims=True)) if causal else tf.softmax(k, axis=-3)
            else:
                if exists(kernel_regularization):
                    q = generalized_kernel(q, projection_matrix=projection_matrix, kernel_function=kernel_regularization)
                    k = generalized_kernel(k, projection_matrix=projection_matrix, kernel_function=kernel_regularization)
                else:
                    q = softmax_kernel(q, projection_matrix=projection_matrix, is_query=True)
                    k = softmax_kernel(k, projection_matrix=projection_matrix, is_query=False)

            # pos emb
            if not gates and exists(position_embedding):
                q = position_embedding(q, q_shape[-2], shape(q)[-1], trainable=trainable, scope='pos_emb_q')
                k = position_embedding(k, k_shape[-2], shape(k)[-1], trainable=trainable, scope='pos_emb_k')

            if causal:
                tensor = causal_linear_attention(q, k, v, mask=q_m, gate=gate, chunks=causal_chunks)
            else:
                tensor = noncausal_linear_attention(q, k, v, mask=q_m)
            mask_tensor = reconstruct_mask(shape(v)[-1], q_m, axis=-1) if exists(mask_query) else None

            if context_projection:
                tensor, mask_tensor = linear(
                        tensor, in_features=shape(tensor)[-2:], out_features=out_features, mask=mask_tensor, bias=projection_bias,
                        lrmul=lrmul, quantization=quantization, quantization_blocks=hidden_features * quantization_blocks // q_shape[-1],
                        weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                        scope='projection_out')

            else:
                tensor, mask_tensor = map(lambda t: tf.reshape(t, shape(t)[:-2] + [shape(t)[-2] * shape(t)[-1]]) if exists(t) else None, (tensor, mask_tensor))

            tensor = tf.reshape(tensor, q_shape[:-1] + [out_features])
            tensors.append(tensor)
            masks_tensor.append(mask_tensor)

        if len(tensors) == 1: tensors, masks_tensor = *tensors, *masks_tensor

    return tensor, mask_tensor, saved_state



def aligned_window(tensor, window_size, tensor_length, pad_values=-1, mask=None):
    b, l, h, d = shape(tensor)
    ratio = tf.cast(l, dtype=tf.float32) / tf.cast(tensor_length, dtype=tf.float32)
    lp, rp = (window_size - 1) // 2, window_size // 2
    padded = tf.pad(tensor, ((0, 0), (lp, rp), (0, 0), (0, 0)), constant_values=pad_values)
    alignment = tf.cast(tf.math.floor(tf.range(tensor_length, dtype=tf.float32) * ratio), dtype=tf.int32)
    indices = alignment[:, None] + tf.range(window_size)[None]
    tensor = tf.gather(padded, indices, axis=1, batch_dims=0)
    if exists(mask):
        mask = tf.pad(mask, ((0, 0), (lp, rp), (0, 0), (0, 0)), constant_values=0.0)
        mask = tf.gather(mask, indices, axis=1, batch_dims=0)
    return tensor, mask


def local_attention(q, k, v, window_size, mask_q=None, mask_k=None, causal=False):
    if not exists(mask_q): mask_q = tf.ones_like(q)
    if not exists(mask_k): mask_k = tf.ones_like(k)

    k, mask_k = aligned_window(k, window_size, shape(q)[1], pad_values=-2.**32, mask=mask_k)     # b q w h d
    v, _ = aligned_window(v, window_size, shape(q)[1], pad_values=-2.**32)     # b q w h o

    mask_attn = tf.reduce_max(mask_q[Ellipsis, None, :, :] * mask_k, axis=-1, keepdims=True)     # b q w h 1
    if causal:
        mask_attn *= tf.cast([1] * ((window_size - 1) // 2) + [1] + [0] * (window_size // 2), mask_q.dtype)[None, None, :, None, None]
    invalidity, invalid_padding = tf.equal(mask_attn, 0), (-2 ** 31) * tf.ones_like(mask_attn)

    score = tf.reduce_sum(q[Ellipsis, None, :, :] * k, axis=-1, keepdims=True)     # b q 1 h d, b q w h d -> b q w h 1
    score = tf.where(invalidity, invalid_padding, score)
    score = tf.where(invalidity, invalid_padding, score - tf.reduce_max(score, axis=-3, keepdims=True))
    score = tf.nn.softmax(score, axis=-3)
    score = tf.where(invalidity, tf.zeros_like(score), score)

    attention = tf.reduce_sum(score * v, axis=-3)     # b q w h 1, b q w h o -> b q h o
    if exists(mask_q): attention *= reconstruct_mask(shape(v)[-1], mask=mask_q, axis=-1)

    return attention


def proximal_attention(
        query,
        key,
        value,
        window_size                     =       7,
        out_features                    =       None,
        hidden_features                 =       None,
        heads                           =       4,
        mask_query                      =       None,
        mask_key                        =       None,
        mask_value                      =       None,
        causal                          =       False,
        position_embedding              =       rotary_embedding,
        saved_state                     =       None,
        projection_bias                 =       True,
        quantization                    =       0.0,
        quantization_blocks             =       8,
        lrmul                           =       1.0,
        query_processes                 =       None,
        key_processes                   =       None,
        value_processes                 =       None,
        component_projection            =       True,
        context_projection              =       True,
        query_weight_function           =       None,
        query_bias_function             =       None,
        key_weight_function             =       None,
        key_bias_function               =       None,
        value_weight_function           =       None,
        value_bias_function             =       None,
        out_weight_function             =       None,
        out_bias_function               =       None,
        trainable                       =       True,
        scope                           =       'prox_attn',
        ):
    with tf.variable_scope(scope):
        query_processes, key_processes, value_processes = process_checker(query_processes, key_processes, value_processes)

        q_shape, k_shape, v_shape, dtype = shape(query), shape(key), shape(value), query.dtype

        if not isinstance(heads, int): raise ValueError("The number of heads must be integer, but given {}".format(type(heads)))

        out_features = int(out_features) if exists(out_features) else q_shape[-1]
        if out_features % heads != 0: raise ValueError("The number of heads must divide out_units evenly, but heads:{} and out_units: {}".format(heads, out_features))

        hidden_features = int(hidden_features) if exists(hidden_features) else out_features
        if hidden_features % heads != 0: raise ValueError("The number of heads must divide hidden_units evenly, but heads:{} and hidden_units: {}".format(heads, hidden_features))

        hidden_depth = hidden_features // heads


        # material projection
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



        processed_q = apply_process(q, query_processes, mask=q_m)
        processed_k = apply_process(k, key_processes, mask=k_m)
        processed_v = apply_process(v, value_processes, mask=v_m)

        tensors, masks_tensor = [], []
        for n, (q, k, v) in enumerate(list(zip(processed_q, processed_k, processed_v))):

            # pos emb
            if exists(position_embedding):
                q = position_embedding(q, q_shape[-2], shape(q)[-1], trainable=trainable, scope='pos_emb_q')
                k = position_embedding(k, k_shape[-2], shape(k)[-1], trainable=trainable, scope='pos_emb_k')

            tensor = local_attention(q, k, v, window_size=window_size, mask_q=q_m, mask_k=k_m, causal=causal)
            mask_tensor = reconstruct_mask(shape(v)[-1], q_m, axis=-1) if exists(mask_query) else None

            if context_projection:
                tensor, mask_tensor = linear(tensor, in_features=shape(tensor)[-2:], out_features=out_features, mask=mask_tensor,
                        bias=projection_bias, lrmul=lrmul, quantization=quantization,
                        quantization_blocks=hidden_features * quantization_blocks // q_shape[-1],
                        weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                        scope='projection_out')
            else:
                tensor, mask_tensor = map(lambda t: tf.reshape(t, shape(t)[:-2] + [shape(t)[-2] * shape(t)[-1]]) if exists(t) else None, (tensor, mask_tensor))

            tensor = tf.reshape(tensor, q_shape[:-1] + [out_features])
            tensors.append(tensor)
            masks_tensor.append(mask_tensor)

        if len(tensors) == 1: tensors, masks_tensor = *tensors, *masks_tensor

    return tensors, masks_tensor, saved_state



def hybrid_fast_attention(
        query,
        key,
        value,
        window_size                     =       7,
        out_features                    =       None,
        hidden_features                 =       None,
        full_heads                      =       4,
        prox_heads                      =       4,
        mask_query                      =       None,
        mask_key                        =       None,
        mask_value                      =       None,
        causal                          =       False,
        gates                           =       True,
        position_embedding              =       rotary_embedding,
        orthogonal_random_features      =       True,
        orthogonal_scaling              =       0,
        kernel_regularization           =       tf.nn.relu,
        saved_state                     =       None,
        projection_bias                 =       True,
        quantization                    =       0.0,
        quantization_blocks             =       8,
        lrmul                           =       1.0,
        query_processes                 =       None,
        key_processes                   =       None,
        value_processes                 =       None,
        component_projection            =       True,
        context_projection              =       True,
        query_weight_function           =       None,
        query_bias_function             =       None,
        key_weight_function             =       None,
        key_bias_function               =       None,
        value_weight_function           =       None,
        value_bias_function             =       None,
        out_weight_function             =       None,
        out_bias_function               =       None,
        trainable                       =       True,
        scope                           =       'prox_attn',
        ):
    with tf.variable_scope(scope):
        query_processes, key_processes, value_processes = process_checker(query_processes, key_processes, value_processes)

        q_shape, k_shape, v_shape, dtype = shape(query), shape(key), shape(value), query.dtype
        if not isinstance(full_heads, int): raise ValueError("The number of full_heads must be integer, but given {}".format(type(full_heads)))
        if not isinstance(prox_heads, int): raise ValueError("The number of prox_heads must be integer, but given {}".format(type(prox_heads)))
        heads = int(full_heads + prox_heads)


        out_features = int(out_features) if exists(out_features) else q_shape[-1]
        if out_features % heads != 0: raise ValueError("The number of heads must divide out_units evenly, but heads:{} and out_units: {}".format(heads, out_features))

        hidden_features = int(hidden_features) if exists(hidden_features) else out_features
        if hidden_features % heads != 0: raise ValueError("The number of heads must divide hidden_units evenly, but heads:{} and hidden_units: {}".format(heads, hidden_features))

        hidden_depth = hidden_features // heads
        theta_random_features = hidden_depth * int(np.ceil(np.log(hidden_depth)))

        gates = gates and causal

        # material projection
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

        if gates:
            gate, gate_m = linear(
                    key, out_features=heads, mask=mask_key, bias=True, lrmul=lrmul, trainable=trainable, scope='projection_gate')
            gate = tf.math.sigmoid(gate)
            if exists(gate_m): gate *= gate_m
            if exists(saved_state):
                gate_saved, gate_m_saved = getattr(saved_state, 'gate', None), getattr(saved_state, 'gate_mask', None)
                gate, gate_m = link_memory(gate, gate_m, tensor_saved=gate_saved, mask_saved=gate_m_saved, distance=distance)
                setattr(saved_state, 'gate', gate)
                setattr(saved_state, 'gate_mask', gate_m)
        else:
            gate = None

        if orthogonal_random_features:
            saved_matrix = tf.get_variable("projection_matrix", shape=[theta_random_features, hidden_depth],
                    dtype=dtype, initializer=tf.initializers.random_normal(0, 1), trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])

            if trainable:
                projection_matrix = gaussian_orthogonal_random_matrix(theta_random_features, hidden_depth,
                        dtype=dtype, scaling=orthogonal_scaling, struct_mode=False)

                def update():
                    with tf.control_dependencies([tf.assign(saved_matrix, projection_matrix)]):
                        return tf.identity(projection_matrix)

                condition = tf.abs(tf.reduce_mean(projection_matrix)) > tf.abs(tf.reduce_mean(saved_matrix))
                projection_matrix = tf.cond(condition, update, lambda: tf.identity(projection_matrix))
            else:
                projection_matrix = saved_matrix

        processed_q = apply_process(q, query_processes, mask=q_m)
        processed_k = apply_process(k, key_processes, mask=k_m)
        processed_v = apply_process(v, value_processes, mask=v_m)

        tensors, masks_tensor = [], []
        for n, (q, k, v) in enumerate(list(zip(processed_q, processed_k, processed_v))):

            with tf.variable_scope('full_heads'):
                q_full, k_full, v_full = q[Ellipsis, :full_heads, :], k[Ellipsis, :full_heads, :], v[Ellipsis, :full_heads, :]

                # orthogonal_random_features & kernel_regularization
                if not orthogonal_random_features:
                    q_full = tf.softmax(q_full, axis=-1)
                    k_full = tf.exp(k_full - tf.reduce_max(k_full, axis=-3, keepdims=True)) if causal else tf.softmax(k_full, axis=-3)
                else:
                    if exists(kernel_regularization):
                        q_full = generalized_kernel(q_full, projection_matrix=projection_matrix, kernel_function=kernel_regularization)
                        k_full = generalized_kernel(k_full, projection_matrix=projection_matrix, kernel_function=kernel_regularization)
                    else:
                        q_full = softmax_kernel(q_full, projection_matrix=projection_matrix, is_query=True)
                        k_full = softmax_kernel(k_full, projection_matrix=projection_matrix, is_query=False)

                # pos emb
                if not gates and exists(position_embedding):
                    q_full = position_embedding(q_full, q_shape[-2], shape(q)[-1], trainable=trainable, scope='pos_emb_q')
                    k_full = position_embedding(k_full, k_shape[-2], shape(k)[-1], trainable=trainable, scope='pos_emb_k')

                if causal:
                    full = causal_linear_attention(q_full, k_full, v_full, mask=q_m[Ellipsis, :full_heads, :], gate=gate, chunks=causal_chunks)
                else:
                    full = noncausal_linear_attention(q_full, k_full, v_full, mask=q_m[Ellipsis, :full_heads, :])

            with tf.variable_scope('prox_heads'):
                q_prox, k_prox, v_prox = q[Ellipsis, full_heads:, :], k[Ellipsis, full_heads:, :], v[Ellipsis, full_heads:, :]

                # pos emb
                if exists(position_embedding):
                    q_prox = position_embedding(q_prox, q_shape[-2], shape(q)[-1], trainable=trainable, scope='pos_emb_q')
                    k_prox = position_embedding(k_prox, k_shape[-2], shape(k)[-1], trainable=trainable, scope='pos_emb_k')

                prox = local_attention(q_prox, k_prox, v_prox, window_size=window_size,
                        mask_q=q_m[Ellipsis, full_heads:, :], mask_k=k_m[Ellipsis, full_heads:, :], causal=causal)

            tensor = tf.concat([full, prox], axis=-2)
            mask_tensor = reconstruct_mask(shape(v)[-1], q_m, axis=-1) if exists(mask_query) else None

            if context_projection:
                tensor, mask_tensor = linear(tensor, in_features=shape(tensor)[-2:], out_features=out_features, mask=mask_tensor,
                        bias=projection_bias, lrmul=lrmul, quantization=quantization,
                        quantization_blocks=hidden_features * quantization_blocks // q_shape[-1],
                        weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                        scope='projection_out')
            else:
                tensor, mask_tensor = map(lambda t: tf.reshape(t, shape(t)[:-2] + [shape(t)[-2] * shape(t)[-1]]) if exists(t) else None, (tensor, mask_tensor))

            tensor = tf.reshape(tensor, q_shape[:-1] + [out_features])
            tensors.append(tensor)
            masks_tensor.append(mask_tensor)

        if len(tensors) == 1: tensors, masks_tensor = *tensors, *masks_tensor

    return tensors, masks_tensor, saved_state



