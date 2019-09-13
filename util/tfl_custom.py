import tensorflow as tf
import numpy as np
from util.helpers import getshape, next_base2, fixed_point
from util.variable import variableFromSettings
from template.misc import S, GLOBAL

# from tensorflow.contrib.framework import broadcast_to
from tensorflow.layers import average_pooling2d
from tensorflow.layers import max_pooling2d
from tensorflow.layers import batch_normalization
from tensorflow.layers import BatchNormalization
from tensorflow.python.ops import init_ops
from tensorflow.layers import dense
from tensorflow.layers import flatten
# import tensorflow_probabilities as tfl
from tensorflow.python.keras.engine.base_layer import unique_layer_name


# from tensorflow.python.util.tf_export import tf_export
# from tensorflow.nn import batch_normalization as batch_normalization_nn


# @tf_export("nn.batch_normalization",overrides=[batch_normalization_nn])
# def batch_normalization_new(**kwargs):
#     print("batch_nn")
#     return batch_normalization_nn(**kwargs)

# GLOBAL["first_layer"] = True
GLOBAL["first_layer"] = False

def batch_normalization(inputs, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer=init_ops.zeros_initializer(), gamma_initializer=init_ops.ones_initializer(), moving_mean_initializer=init_ops.zeros_initializer(), moving_variance_initializer=init_ops.ones_initializer(), beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, training=False, trainable=True, name=None, reuse=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None, virtual_batch_size=None, adjustment=None):
    layer = BatchNormalization( axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, renorm=renorm, renorm_clipping=renorm_clipping, renorm_momentum=renorm_momentum, fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size, adjustment=adjustment, name=name, _reuse=reuse, _scope=name)
    res = layer.apply(inputs, training=training)

    if not S("batch_norm.transform"):
        return res

    # get moving mean and variance
    moving_mean, moving_variance = layer.moving_mean, layer.moving_variance
    beta_offset, gamma_scale = layer.beta, layer.gamma

    if GLOBAL["first_layer"]:
        GLOBAL["first_layer"] = False
    else:
        pass
        # print("reformulate batchnorm")

        # apply transformation
        # --------------------
        # moving_mean = variableFromSettings([],hiddenVar=moving_mean)[0]
        # moving_variance = variableFromSettings([],hiddenVar=moving_variance)[0]
        # beta_offset = variableFromSettings([],hiddenVar=beta_offset)[0]
        # gamma_scale = variableFromSettings([],hiddenVar=gamma_scale)[0]

        # apply transformation (no var)
        # --------------------
        # sample_size = S("binom.sample_size")
        # S("binom.sample_size",set=sample_size*4)
        # gamma_scale = gamma_scale/tf.sqrt(moving_variance+layer.epsilon)
        # gamma_scale = variableFromSettings([],hiddenVar=gamma_scale/tf.sqrt(moving_variance+layer.epsilon))[0]
        # moving_variance = 0*moving_variance+1
        # moving_variance = tf.ones_like(moving_variance)
        # S("binom.sample_size",set=sample_size)

        # moving_variance = 1.0/variableFromSettings([],hiddenVar=1.0/moving_variance)[0]
        # moving_mean = fixed_point(moving_mean,8)
        # moving_mean, _ = variableFromSettings([],hiddenVar=moving_mean)
        # moving_variance = next_base2(moving_variance, strict_positive=True)
        # moving_variance = 2**tf.ceil(tf.log(tf.maximum(tf.abs(moving_variance),0))/tf.log(2.0))
    # tf.summary.histogram("bn_mean",moving_mean)
    # tf.summary.histogram("bn_var",moving_variance)

    # set moving mean and variance
    layer.moving_mean, layer.moving_variance = moving_mean, moving_variance
    layer.beta, layer.gamma = beta_offset, gamma_scale

    # reapply
    res = layer.apply(inputs, training=training)

    return res

def bernoulli_fc(x,out_dim, name=None):
    shape = [np.prod(getshape(x))] +  [out_dim]
    weight_s, _ = variableFromSettings(shape)
    return tf.matmul(tf.layers.flatten(x), weight_s, name=None)

def conv2d(inputs, filters, kernel_size, strides, padding, name="conv2d", rate=1, scope=None, normalizer_fn=None, return_preact=False, **args):

    activation = False
    if "activation_fn" in args:
        activation_fn = args["activation_fn"]
        del args["activation_fn"]
        activation = True

    if "use_bias" in args and args["use_bias"]:
        assert not args["use_bias"]
        del args["use_bias"]

    if "kernel_initializer" in args:
        del args["kernel_initializer"]

    if "data_format" in args:
        args["data_format"] = "NCHW" if args["data_format"] == "channels_first" else "NHWC"
    else:
        args["data_format"] = "NHWC"

    # kernel
    if not isinstance(kernel_size, list):
        kernel_size = [kernel_size]*2

    # data_format dependent kernel_size, strides, dilations
    if args["data_format"] == "NHWC":
        num_channels = getshape(inputs)[-1]
        if not isinstance(strides, list):
            strides = [1]+[strides]*2+[1]
        dilations = [1,rate, rate,1]
    else:
        num_channels = getshape(inputs)[0]
        if not isinstance(strides, list):
            strides = [1,1]+[strides]*2
        dilations = [1,1]+[rate, rate]

    # get filter
    with tf.variable_scope(unique_layer_name(name,zero_based=True)) as vc:
        weight_s, weight_p = variableFromSettings(kernel_size+[num_channels,filters])

        # conv with sampled filter
        conv_args = {"input":inputs, "strides":strides, "padding":padding, "dilations":dilations, **args}
        conv = tf.nn.conv2d(filter=weight_s, name=name, **conv_args)

    if normalizer_fn is not None:
        conv = normalizer_fn(conv,name="bn")

    # check if an activation needs some work beforehand
    if activation and activation_fn is not None:
        if "activation_prepare" in GLOBAL:
            preact = GLOBAL["activation_prepare"](weight_s, weight_p, inputs, conv, conv_args)
            if return_preact:
                return preact

        conv = activation_fn(conv)

    # just return that
    return conv

