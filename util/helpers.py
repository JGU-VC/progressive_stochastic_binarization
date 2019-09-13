import tensorflow as tf
import numpy as np
from template.misc import S, GLOBAL, print_info as print
if S("binom.probability_mode") == "tfp":
    import tensorflow_probability as tfp


# -------------- #
# tensor helpers #
# -------------- #

# get shape (excluding input shape)
def getshape(x):
    return x.get_shape().as_list()[1:]

# pass gradient around non-diferentiable function
def pass_gradient(x,backward_fn, forward_fn=lambda x: x, name=None):
    fnx = forward_fn(x)
    return tf.add(fnx, tf.stop_gradient(backward_fn(x) - fnx),name=name)

# guess picture shape and reshape
def to_picture_shape(input):
    current_shape = input.get_shape().as_list()[1:]
    current_dim = np.prod(current_shape)

    shape = None
    for i in range(256,3,-1):
        if current_dim % (i*i) == 0:
            shape = [-1, i, i, int(current_dim / (i*i))]
            break
    if shape is None:
        raise ValueError("This does not seem to be a picture")
    input = tf.reshape(input, shape)
    return input

# evaluate tensor by name
def get_tensor(name, sess):
    if not isinstance(name, list):
        name = [name]
    return sess.run([sess.graph.get_tensor_by_name(n) for n in name])

# def guess_layer_from_name(name):
#     name = name.split("_")
#     if len(name)==2:
#         return 0
#     else:
#         return int(name[-1].split(":")[0])

def fixed_point(x,bits,max=1,min=0):
    with tf.name_scope('fixed_point'):
        if max!=1 or min !=0:
            x = (x - min)/(max-min)
        x = tf.clip_by_value(x,0,1,name='clip')

        # x_fixed = tf.round(x * 2**bits)/2**bits
        # x_fixed = tf.floor(x * 2**bits)/2**bits
        # x_fixed = tf.floor(x * 2**bits + 0.5)/2**bits
        # x_fixed = tf.where(x<0,tf.floor(x*2**bits),tf.ceil(x*2**bits))/2**bits
        x_fixed = tf.floor(x * 2**bits)/(2**bits-1)
        x = pass_gradient(x,lambda x: x_fixed, name='fixed_point')

        if max!=1 or min !=0:
            x = x*(max-min) + min
        return x

def next_base2(x, strict_positive=False, stochastic=False, min=1e-8, binom_n=64):
    with tf.name_scope('next_base2'):
        x_start = x
        if strict_positive:
            sign = 1
        else:
            sign = tf.sign(x)
        if stochastic:
            # x_next_base2 = tf.floor(tf.log(tf.abs(x+eps))/tf.log(2.0))
            x_next_base2 = tf.floor(tf.log(tf.maximum(tf.abs(x),min))/tf.log(2.0))
            x_perc_missing = tf.abs(x)/2**x_next_base2-1
            # w_add = where_binarize[0,1]->{0,1}(x+exs)
            print("next_base2: stochastic-mode '"+str(stochastic)+"'")
            if stochastic == "binomial" or stochastic == "binom":
                memory_size = binom_n
                w_add = sample_binomial(x_perc_missing,memory_size,S('binom.log_eps'))/memory_size
                tf.summary.histogram("w_add",w_add)
            else:
                w_add = tf.where(tf.random.uniform(x.get_shape().as_list()) <= x_perc_missing, tf.ones_like(x), tf.zeros_like(x))
            x_next_base2 += w_add
        else:
            x_next_base2 = tf.ceil(tf.log(tf.maximum(tf.abs(x),min))/tf.log(2.0))
        return pass_gradient(x_start, lambda x: sign*2**x_next_base2, name='next_base2')

def sample_binomial(p,n,eps=S('binom.log_eps')):
    # sample from a binomial distribution
    if S("binom.probability_mode") == "tfp":
        P = tf.stack([p,1.0-p],axis=-1)
        weight_binom = tfp.distributions.Multinomial(total_count=n,probs=P).sample()[...,0]
        # weight_binom = tfp.distributions.Binomial(total_count=n,probs=p).sample()
        weight_binom = tf.cast(weight_binom,tf.float32)
    elif S("binom.probability_mode") == "gumbel":
        with tf.variable_scope("p"):
            # p = weight_p
            p = tf.clip_by_value(p, 0.0, 1.0)
            P = tf.stack([binomialCoeff(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)],axis=-1)

            # reduces numerical instabilities
            P = tf.clip_by_value(P,eps,1.0)
            gumbel = -tf.log(tf.maximum(-tf.log(tf.maximum(tf.random.uniform(P.get_shape()),eps)),eps))

            # gumbel = -tf.log(-tf.log(tf.random.uniform(P.get_shape())))
            # tf.summary.histogram("binom_p",p)
            # tf.summary.histogram("binom_P",P)
            # tf.summary.histogram("binom_logP",tf.log(P))
        weight_binom = tf.argmax(tf.log(P)+gumbel,axis=-1)
        weight_binom = tf.cast(weight_binom,tf.float32)
    elif S("binom.probability_mode") == "gumbel_log":
        with tf.variable_scope("p"):
            # p = weight_p
            p = tf.clip_by_value(p, eps, 1.0-eps)
            logP = tf.stack([np.log(binomialCoeff(n,k))+k*tf.log(p)+(n-k)*tf.log(1-p) for k in range(n+1)],axis=-1)

            # reduces numerical instabilities
            gumbel = -tf.log(tf.maximum(-tf.log(tf.maximum(tf.random.uniform(logP.get_shape()),eps)),eps))

        weight_binom = tf.argmax(logP+gumbel,axis=-1)
        weight_binom = tf.cast(weight_binom,tf.float32)

    if S("binom.gradient_correction") == "pass":
        weight_binom = pass_gradient(p, lambda p: weight_binom, lambda p: n*p)
    elif S("binom.gradient_correction") == "gumbel":
        weight_binom = pass_gradient(p, lambda p: weight_binom, lambda p: tf.squeeze(tf.batch_gather(P,tf.cast(tf.expand_dims(weight_binom,-1),tf.int32))))
    else:
        raise ValueError("Gradient not defined for tf.cast. TODO")
    return weight_binom

def binomialCoeff(n, k):
    result = 1
    for i in range(1, k+1):
        result = result * (n-i+1) / i
    return result

def compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
    Src:
        https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/init_ops.py
    Args:
        shape: Integer shape tuple or TF tensor shape.
    Returns:
        A tuple of scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


# source: tensorflow/models/official/resnet
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
    base_lr=0.1, warmup=False):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    if warmup:
      warmup_steps = int(batches_per_epoch * 5)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
      return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
    return lr

  return learning_rate_fn


# src: https://github.com/tensorflow/tensorflow/issues/1823
def optimistic_restore(save_file, graph=tf.get_default_graph()):
    print("optimistic_restore")
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])    
    restore_vars = []    
    for var_name, saved_var_name in var_names:            
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    return opt_saver
