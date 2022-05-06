

import numpy as np
import tensorflow as tf

"""
https://explosion.ai/blog/deep-learning-formula-nlp
embed -> encode -> attend -> predict
"""
def batch_normalization(x, training, name):
    # with tf.variable_scope(name, reuse=)
    bn_train = tf.layers.batch_normalization(x, training=True, reuse=None, name=name)
    bn_inference = tf.layers.batch_normalization(x, training=False, reuse=True, name=name)
    z = tf.cond(training, lambda: bn_train, lambda: bn_inference)
    return z


#### Step 1
def embed(x, size, dim, seed=0, flatten=False, reduce_sum=False):
    # std = np.sqrt(2 / dim)
    std = 0.001
    minval = -std
    maxval = std
    emb = tf.Variable(tf.random_uniform([size, dim], minval, maxval, dtype=tf.float32, seed=seed))
    # None * max_seq_len * embed_dim