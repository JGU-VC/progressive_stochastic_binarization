import tensorflow as tf
import numpy as np
from template.misc import S, GLOBAL
# from model.util import layer_from_name


class RayGrad(tf.train.AdamOptimizer):

    def __init__(self, learning_rate=0.1, epsilon=None, use_locking=False):
        super(RayGrad, self).__init__(learning_rate, epsilon=epsilon, use_locking=use_locking)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.memory_size = S("optimizer.memory_size")
        self.loss_collect_last = S("optimizer.collect_last")

    def minimize(self, loss, global_step=None, var_list=None, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):

        # compute (meaned) gradients for a batch
        grads_and_vars = self.compute_gradients(loss, var_list=var_list, aggregation_method=aggregation_method, colocate_gradients_with_ops=colocate_gradients_with_ops, grad_loss=grad_loss)

        # check if any trainable variables provided
        for g,v in grads_and_vars:
            if g is None:
                print("Gradient of '"+v.name+"' is 'None'. Ignoring")
        grads_and_vars = [(g,v) for g,v in grads_and_vars if g is not None]

        # default adam does:
        # return self.apply_gradients(grads_and_vars, global_step=global_step, name=name)

        # get all trainable variables
        variables = [v for g,v in grads_and_vars]

        # create a copy of all trainable variables with `0` as initial values
        with tf.name_scope("optimizer"):
            gradient_sum = [tf.get_variable(v.name.replace(":0","_sum"), initializer=tf.zeros_like(v.initialized_value()),trainable=False) for v in variables]

        def capacity_gradient(grad_sum,grad,name,var):
            if "hiddenWeight" in name and "weight_gradient" in GLOBAL:
                return GLOBAL["weight_gradient"](grad_sum,grad,var)
            return grad_sum + grad

        with tf.control_dependencies([GLOBAL["memory_step"]]):

            # collect the batch gradient into accumulated vars
            gradient_sum_update = [
                gs.assign(
                    tf.where(GLOBAL["memory_step"]>0,
                        capacity_gradient(gs,g,v.name,v),
                        g)
                    )
                for gs,(g,v) in zip(gradient_sum,grads_and_vars)
            ]

            with tf.control_dependencies(gradient_sum_update):
                train_step = tf.cond(GLOBAL["memory_step"] >= S("optimizer.memory_size")-1,
                        true_fn=lambda: self.apply_gradients([(gs/S("optimizer.memory_size"), v) for gs,(g,v) in zip(gradient_sum,grads_and_vars)], global_step),
                        false_fn=lambda: tf.no_op())

        return train_step



# optimizer = tf.train.AdamOptimizer
optimizer = RayGrad
# optimizer = NoOptimizer
