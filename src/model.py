

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