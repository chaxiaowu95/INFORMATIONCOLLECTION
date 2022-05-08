

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
    out = tf.nn.embedding_lookup(emb, x)
    if flatten:
        out = tf.layers.flatten(out)
    if reduce_sum:
        out = tf.reduce_sum(out, axis=1)
    return out


def embed_subword(x, size, dim, sequence_length, seed=0, mask_zero=False, maxlen=None):
    # std = np.sqrt(2 / dim)
    std = 0.001
    minval = -std
    maxval = std
    emb = tf.Variable(tf.random_uniform([size, dim], minval, maxval, dtype=tf.float32, seed=seed))
    # None * max_seq_len * max_word_len * embed_dim
    out = tf.nn.embedding_lookup(emb, x)
    if mask_zero:
        # word_len: None * max_seq_len
        # mask: shape=None * max_seq_len * max_word_len
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.cast(mask, tf.float32)
        out = out * mask
    # None * max_seq_len * embed_dim
    # according to facebook subword paper, it's sum
    out = tf.reduce_sum(out, axis=2)
    return out


def word_dropout(x, training, dropout=0, seed=0):
    # word dropout (dropout the entire embedding for some words)
    """
    tf.layers.Dropout doesn't work as it can't switch training or inference
    """
    if dropout > 0:
        input_shape = tf.shape(x)
        noise_shape = [input_shape[0], input_shape[1], 1]
        x = tf.layers.Dropout(rate=dropout, noise_shape=noise_shape, seed=seed)(x, training=training)
    return x


#### Step 2
def fasttext(x):
    return x


def textcnn(x, num_filters=8, filter_sizes=[2, 3], bn=False, training=False,
            timedistributed=False, scope_name="textcnn", reuse=False):
    # x: None * step_dim * embed_dim
    conv_blocks = []
    for i, filter_size in enumerate(filter_sizes):
        scope_name_i = "%s_textcnn_%s"%(str(scope_name), str(filter_size))
        with tf.variable_scope(scope_name_i, reuse=reuse):
            if timedistributed:
                input_shape = tf.shape(x)
                step_dim = input_shape[1]
                embed_dim = input_shape[2]
                x = tf.transpose(x, [0, 2, 1])
                # None * embed_dim * step_dim
                x = tf.reshape(x, [input_shape[0] * embed_dim, step_dim, 1])
                conv = tf.layers.conv1d(
                    input=x,
                    filters=1,
                    kernel_size=filter_size,
                    padding="same",
                    activation=None,
                    strides=1,
                    reuse=reuse,
                    name=scope_name_i)
                conv = tf.reshape(conv, [input_shape[0], embed_dim, step_dim])
                conv = tf.transpose(conv, [0, 2, 1])
            else:
                conv = tf.layers.conv1d(
                    inputs=x,
                    filters=num_filters,
                    kernel_size=filter_size,
                    padding="same",
                    activation=None,
                    strides=1,
                    reuse=reuse,
                    name=scope_name_i)
            if bn:
                conv = tf.layers.BatchNormalization()(conv, training)
            conv = tf.nn.relu(conv)
            conv_blocks.append(conv)
    if len(conv_blocks) > 1:
        z = tf.concat(conv_blocks, axis=-1)
    else:
        z = conv_blocks[0]
    return z


def textrnn(x, num_units, cell_type, sequence_length, num_layers=1, mask_zero=False, scope_name="textrnn", reuse=False):
    for i in range(num_layers):