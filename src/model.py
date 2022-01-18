

import time
import numpy as np
import pandas as pd
import tensorflow as tf

import utils
from metrics import ndcg, calc_err
from tf_common.nn_module import resnet_block, dense_block
from tf_common.nadam import NadamOptimizer


class BaseRankModel(object):

    def __init__(self, model_name, params, logger, training=True):
        self.model_name = model_name
        self.params = params
        self.logger = logger
        utils._makedirs(self.params["offline_model_dir"], force=training)

        self._init_tf_vars()
        self.loss, self.num_pairs, self.score, self.train_op = self._build_model()

        self.sess, self.saver = self._init_session()


    def _init_tf_vars(self):
        with tf.name_scope(self.model_name):
            #### input for training and inference
            self.feature = tf.placeholder(tf.float32, shape=[None, self.params["feature_dim"]], name="feature")
            self.training = tf.placeholder(tf.bool, shape=[], name="training")
            #### input for training
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
            self.sorted_label = tf.placeholder(tf.float32, shape=[None, 1], name="sorted_label")
            self.qid = tf.placeholder(tf.float32, shape=[None, 1], name="qid")
            #### vars for training
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.params["init_lr"], self.global_step,
                                                            self.params["decay_steps"], self.params["decay_rate"])
            self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")


    def _build_model(self):
        return None, None, None, None


    def _score_fn_inner(self, x, reuse=False):
        # deep
        hidden_units = [self.params["fc_dim"] * 4, self.params["fc_dim"] * 2, self.params["fc_dim"]]
        dropouts = [self.params["fc_dropout"]] * len(hidden_units)
        out = dense_block(x, hidden_units=hidden_units, dropouts=dropouts, densenet=False, reuse=reuse,
                          training=self.training, seed=self.params["random_seed"])
        # score
        score = tf.layers.dense(out, 1, activation=None,
                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.params["random_seed"]))

        return score


    def _score_fn(self, x, reuse=False):
        # https://stackoverflow.com/questions/45670224/why-the-tf-name-scope-with-same-name-is-different
        with tf.name_scope(self.model_name+"/"):
            score = self._score_fn_inner(x, reuse)
            # https://stackoverflow.com/questions/46980287/output-node-for-tensorflow-graph-created-with-tf-layers
            # add an identity node to output graph
            score = tf.identity(score, "score")

        return score


    def _jacobian(self, y_flat, x):
        """
        https://github.com/tensorflow/tensorflow/issues/675
        for ranknet and lambdarank
        """
        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=self.batch_size),
        ]

        _, jacobian = tf.while_loop(
            lambda j, _: j < self.batch_size,
            lambda j, result: (j + 1, result.write(j, tf.gradients(y_flat[j], x))),
            loop_vars)

        return jacobian.stack()