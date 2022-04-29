

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
        self._beta2 = beta2
        self._epsilon = epsilon
        self._schedule_decay = schedule_decay
        # momentum cache decay
        self._momentum_cache_decay = tf.cast(0.96, tf.float32)
        self._momentum_cache_const = tf.pow(self._momentum_cache_decay, 1. * schedule_decay)

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._schedule_decay_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None
        self._iterations = None
        self._m_schedule = None

        # Created in SparseApply if needed.
        self._updated_lr = None


    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
        self._schedule_decay_t = ops.convert_to_tensor(self._schedule_decay, name="schedule_decay")

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._iterations is None
        if not create_new and context.in_graph_mode():
            create_new = (self._iterations.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1,
                                                            name="beta1_power",
                                                            trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2,
                                                            name="beta2_power",
                                                            trainable=False)
                self._iterations = variable_scope.variable(0.,
                                                           name="iterations",
                                                           trainable=False)
                self._m_schedule = variable_scope.variable(1.,
                                                           name="m_schedule",
                                                           trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _get_momentum_cache(self, schedule_decay_t, t):
        return tf.pow(self._momentum_cache_decay, t * schedule_decay_t)
        # return beta1_t * (1. - 0.5 * (tf.pow(self._momentum_cache_decay, t * schedule_decay_t)))


    """very slow
    we simply use the nadam update rule without warming momentum schedule
    def _apply_dense(self, grad, var):
        t = math_ops.cast(self._iterations, var.dtype.base_dtype) + 1.
        m_schedule = math_ops.cast(self._m_schedule, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        schedule_decay_t = math_ops.cast(self._schedule_decay_t, var.dtype.base_dtype)

        # Due to the recommendations in [2], i.e. warming momentum schedule
        # see keras Nadam
        momentum_cache_t = self._get_momentum_cache(beta1_t, schedule_decay_t, t)
        momentum_cache_t_1 = self._get_momentum_cache(beta1_t, schedule_decay_t, t+1.)
        m_schedule_new = m_schedule * momentum_cache_t
        m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1

        # the following equations given in [1]
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + (1. - beta1_t) * grad, use_locking=self._use_locking)
        g_prime = grad / (1. - m_schedule_new)
        m_t_prime = m_t / (1. - m_schedule_next)
        m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, beta2_t * v + (1. - beta2_t) * tf.square(grad), use_locking=self._use_locking)
        v_t_prime = v_t / (1. - tf.pow(beta2_t, t))

        var_update = state_ops.assign_sub(var,
                                      lr_t * m_t_bar / (tf.sqrt(v_t_prime) + epsilon_t),
                                      use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])
    """
    # nadam update rule without warming momentum schedule
    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        return training_ops.apply_adam(
            var,
            m,
            v,
            math_ops.cast(self._beta1_power, var.dtype.base_dtype),
            math_ops.cast(self._beta2_power, var.dtype.base_dtype),
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
            use_nesterov=True).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        return training_ops.resource_apply_adam(
            var.handle,
            m.handle,
            v.handle,
            math_ops.cast(self._beta1_power, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_power, grad.dtype.base_dtype),
            math_ops.cast(self._lr_t, grad.dtype.base_dtype),