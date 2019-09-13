import tensorflow as tf
from settings import SETTINGS
from template.misc import GLOBAL, S

def lossfn(net_out, data, labels_one_hot, mode):
    with tf.name_scope('cross_entropy'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_one_hot, logits=net_out)
        tf.summary.scalar("loss", loss)

    with tf.name_scope('regularization'):
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_variables) > 0:
            reg = tf.reduce_mean(reg_variables, name="regularization_loss")
            tf.summary.scalar('regularization', reg)
        else:
            reg = 0

    with tf.name_scope('total_loss'):
        if reg != 0:
            total = loss + S("util.variable.regularizer.weight",alt=1.0)*reg
            tf.summary.scalar("total_loss", total)
            return total

    return loss
    # with tf.name_scope('cross_entropy'):

    #     # eval_step_increase = GLOBAL["eval_step_increase"]
    #     # eval_step = GLOBAL["eval_step"]
    #     # memory_step = GLOBAL["memory_step"]

    #     # if mode == tf.estimator.ModeKeys.TRAIN:
    #     #     control_deps = [eval_step_increase, eval_step, memory_step]
    #     # elif mode == tf.estimator.ModeKeys.EVAL:
    #     #     control_deps = [eval_step, memory_step]
    #     # with tf.control_dependencies(control_deps):
    #     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_one_hot, logits=net_out)

    #     # global_step = tf.train.get_or_create_global_step()
    #     # loss = tf.cast((global_step%S("optimizer.memory_size"))/S("optimizer.memory_size"),dtype=tf.float32)*loss

    #     if mode == tf.estimator.ModeKeys.TRAIN:
    #         tf.summary.scalar('loss', loss)
    #     elif mode == tf.estimator.ModeKeys.EVAL:
    #         tf.add_to_collection("SUMMARIES_VALIDATION", tf.summary.scalar('loss',loss, collections="SUMMARIES_VALIDATION"))

    # if S('util.variable.regularizer.type'):
    #     reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #     reg = tf.reduce_mean(reg_variables, name="regularization_loss")
    #     tf.summary.scalar('regularization', reg)
    #     total = loss + 0.01*reg
    #     tf.summary.scalar("total_loss", total)
    #     loss = total


    # return loss
