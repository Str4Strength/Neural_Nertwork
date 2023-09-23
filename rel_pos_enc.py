import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *



def rel_pos_emb(
        l_q,
        l_k,
        depth,
        max_rel_pos,
        cache=False,
        trainable=True,
        scope='rel_pos_emb',
        ):
    with tf.variable_scope(scope):
        # rel_pos_mat
        if not cache:
            range_q = tf.range(l_q)
            range_k = tf.range(l_k)
            distance_mat = range_k[None] - range_q[:, None]
        else:
            distance_mat = tf.range(-l_k+1, 1, 1)[None]
        distance_mat_clipped = tf.clip_by_value(distance_mat, -max_rel_pos, max_rel_pos)
        rel_pos_mat = distance_mat_clipped + max_rel_pos

        vocab_size = max_rel_pos * 2 + 1
        embedding_table = tf.get_variable("embeddings", [vocab_size, depth], initializer=tf.zeros_initializer,
                trainable=False)
        embeddings = tf.gather(embedding_table, rel_pos_mat)
        return embeddings



