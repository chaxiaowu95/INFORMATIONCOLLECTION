

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class NadamOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 schedule_decay=0.004, use_locking=False, name="Nadam"):
        super(NadamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1