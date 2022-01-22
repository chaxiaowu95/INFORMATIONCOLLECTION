

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


    def _get_derivative(self, score, Wk, lambda_ij, x):
        """
        for ranknet and lambdarank
        :param score:
        :param Wk:
        :param lambda_ij:
        :return:
        """
        # dsi_dWk = tf.map_fn(lambda s: tf.gradients(s, [Wk])[0], score) # do not work
        # dsi_dWk = tf.stack([tf.gradients(si, x)[0] for si in tf.unstack(score, axis=1)], axis=2) # do not work
        dsi_dWk = self._jacobian(score, Wk)
        dsi_dWk_minus_dsj_dWk = tf.expand_dims(dsi_dWk, 1) - tf.expand_dims(dsi_dWk, 0)
        shape = tf.concat(
            [tf.shape(lambda_ij), tf.ones([tf.rank(dsi_dWk_minus_dsj_dWk) - tf.rank(lambda_ij)], dtype=tf.int32)],
            axis=0)
        grad = tf.reduce_mean(tf.reshape(lambda_ij, shape) * dsi_dWk_minus_dsj_dWk, axis=[0, 1])
        return tf.reshape(grad, tf.shape(Wk))


    def _get_train_op(self, loss):
        """
        for model that gradient can be computed with respect to loss, e.g., LogisticRegression and RankNet
        """
        with tf.name_scope("optimization"):
            if self.params["optimizer_type"] == "nadam":
                optimizer = NadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                           beta2=self.params["beta2"], epsilon=1e-8,
                                           schedule_decay=self.params["schedule_decay"])
            elif self.params["optimizer_type"] == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                   beta2=self.params["beta2"], epsilon=1e-8)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 1})
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # max_to_keep=None, keep all the models