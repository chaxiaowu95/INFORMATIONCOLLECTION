

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