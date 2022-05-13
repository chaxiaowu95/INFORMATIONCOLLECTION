

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
        scope_name_i = "%s_textrnn_%s_%s_%s" % (str(scope_name), cell_type, str(i), str(num_units))
        with tf.variable_scope(scope_name_i, reuse=reuse):
            if cell_type == "gru":
                cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
            elif cell_type == "lstm":
                cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
            if mask_zero:
                x, _ = tf.nn.dynamic_rnn(cell_fw, x, dtype=tf.float32, sequence_length=sequence_length, scope=scope_name_i)
            else:
                x, _ = tf.nn.dynamic_rnn(cell_fw, x, dtype=tf.float32, sequence_length=None, scope=scope_name_i)
    return x


def textbirnn(x, num_units, cell_type, sequence_length, num_layers=1, mask_zero=False, scope_name="textbirnn", reuse=False):
    for i in range(num_layers):
        scope_name_i = "%s_textbirnn_%s_%s_%s" % (str(scope_name), cell_type, str(i), str(num_units))
        with tf.variable_scope(scope_name_i, reuse=reuse):
            if cell_type == "gru":
                cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
                cell_bw = tf.nn.rnn_cell.GRUCell(num_units)
            elif cell_type == "lstm":
                cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
                cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
            if mask_zero:
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, x, dtype=tf.float32, sequence_length=sequence_length, scope=scope_name_i)
            else:
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, x, dtype=tf.float32, sequence_length=None, scope=scope_name_i)
            x = tf.concat([output_fw, output_bw], axis=-1)
    return x



def encode(x, method, params, sequence_length=None, mask_zero=False, scope_name="encode", reuse=False):
    """
    :param x: shape=(None,seqlen,dim)
    :param params:
    :return: shape=(None,seqlen,dim)
    """
    dim_f = params["embedding_dim"]
    dim_c = len(params["cnn_filter_sizes"]) * params["cnn_num_filters"]
    dim_r = params["rnn_num_units"]
    dim_b = params["rnn_num_units"] * 2
    out_list = []
    params["encode_dim"] = 0
    for m in method.split("+"):
        if m == "fasttext":
            z = fasttext(x)
            out_list.append(z)
            params["encode_dim"] += dim_f
        elif m == "textcnn":
            z = textcnn(x, num_filters=params["cnn_num_filters"], filter_sizes=params["cnn_filter_sizes"],
                        timedistributed=params["cnn_timedistributed"], scope_name=scope_name, reuse=reuse)
            out_list.append(z)
            params["encode_dim"] += dim_c
        elif m == "textrnn":
            z = textrnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"],
                        sequence_length=sequence_length, mask_zero=mask_zero, scope_name=scope_name, reuse=reuse)
            out_list.append(z)
            params["encode_dim"] += dim_r
        elif method == "textbirnn":
            z = textbirnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"],
                          sequence_length=sequence_length, mask_zero=mask_zero, scope_name=scope_name, reuse=reuse)
            out_list.append(z)
            params["encode_dim"] += dim_b
    z = tf.concat(out_list, axis=-1)
    return z


def attention(x, feature_dim, sequence_length=None, mask_zero=False, maxlen=None, epsilon=1e-8, seed=0,
              scope_name="attention", reuse=False):
    input_shape = tf.shape(x)
    step_dim = input_shape[1]
    # feature_dim = input_shape[2]
    x = tf.reshape(x, [-1, feature_dim])
    """
    The last dimension of the inputs to `Dense` should be defined. Found `None`.

    cann't not use `tf.layers.Dense` here
    eij = tf.layers.Dense(1)(x)

    see: https://github.com/tensorflow/tensorflow/issues/13348
    workaround: specify the feature_dim as input
    """
    with tf.variable_scope(scope_name, reuse=reuse):
        eij = tf.layers.dense(x, 1, activation=tf.nn.tanh,
                              kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                              reuse=reuse,
                              name=scope_name)
    eij = tf.reshape(eij, [-1, step_dim])
    a = tf.exp(eij)

    # apply mask after the exp. will be re-normalized next
    if mask_zero:
        # None * step_dim
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.cast(mask, tf.float32)
        a = a * mask

    # in some cases especially in the early stages of training the sum may be almost zero
    a /= tf.cast(tf.reduce_sum(a, axis=1, keep_dims=True) + epsilon, tf.float32)

    a = tf.expand_dims(a, axis=-1)
    return a

