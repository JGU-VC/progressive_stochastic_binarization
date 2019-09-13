import tensorflow as tf
from util.helpers import to_picture_shape, pass_gradient, next_base2, fixed_point
from template.misc import S, GLOBAL, get_distinct_colors
import numpy as np
from util.activations import relu, relu_mean_with_bias, prelu, time_relu, time_relu_sim, time_relu_sim_prepare
from util.tfl import tfl


activation = relu

def network_inner(data, labels_one_hot, mode):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    id = lambda net,name=None: net
    GLOBAL["weight_counter"] = 0

    print("is_training:"+str(is_training))
    batch_normalization = lambda net,name=None: tfl.batch_normalization(net,name=name,reuse=is_training, training=is_training)

    numclasses = GLOBAL["dataset"].num_classes()
    data = to_picture_shape(data)
    net = data
    use_bias = False

    # stack convs
    for i, channels in enumerate(S("model.resnet.conv_blocks")):
        with tf.variable_scope("conv"+str(i)):
            net = tfl.conv2d(net, 64, 3, strides=2, padding="SAME", use_bias=use_bias)
            net = batch_normalization(net)
            net = activation(net)

    # end
    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
    tf.summary.histogram("pre_fc",net)
    if S("model.resnet.last_layer_real"):
        net = tfl.dense( net, numclasses)
    else:
        net = tfl.conv2d( net, numclasses, 1, strides=1, padding="SAME", use_bias=use_bias)
    net = tf.reshape(net, [-1, numclasses])
    return net

scopes_itself = False
network = network_inner
