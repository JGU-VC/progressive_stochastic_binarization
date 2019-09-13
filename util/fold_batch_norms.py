# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Logic to fold batch norm into preceding convolution or FC layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from template.misc import print_info, bcolors, S
from util.helpers import next_base2, fixed_point
print = lambda *args: print_info(*args,color=bcolors.OKBLUE)

import re
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import graph_matcher
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from util.variable import variableFromSettings
from tensorflow.contrib.quantize.python.fold_batch_norms import _BatchNormMatch, _ComputeBatchNormCorrections, _IsValidUnfusedBatchNorm, _FindFusedBatchNorms
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.util import compat
import tensorflow as tf
import numpy as np




# We use matched_layer_set to ensure that layers aren't matched multiple times.
matched_layer_set = set()



def FoldBatchNorms(graph, is_training, freeze_batch_norm_delay=None):
  """Finds batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, true if training.
    freeze_batch_norm_delay: How many steps to wait before freezing moving mean
      and variance and using them for batch normalization. This value is used
      only when is_training is True.
  Raises:
    ValueError: When batch norm folding fails.
  """
  _FoldFusedBatchNorms( graph, is_training, freeze_batch_norm_delay=freeze_batch_norm_delay)
  _FoldUnfusedBatchNorms( graph, is_training=is_training, freeze_batch_norm_delay=freeze_batch_norm_delay)
  _RedoRestFilters(graph)
  _RedoRestBatchnorms(graph, is_training)
  _RedoRestBias(graph)
  _RedoRestAvgPool(graph)

def _FoldFusedBatchNorms(graph, is_training, freeze_batch_norm_delay):
  """Finds fused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, true if training.
    freeze_batch_norm_delay: How many steps to wait before freezing moving mean
      and variance and using them for batch normalization.

  Raises:
    ValueError: When batch norm folding fails.
  """

  matches = list(_FindFusedBatchNorms(graph))
  print("Folding",len(matches),"FusedBatchNorms")
  for match in matches:
    scope, sep, _ = match.layer_op.name.rpartition('/')
    # Make sure new ops are added to `graph` and put on the same device as
    # `bn_op`. The '/' (i.e. `sep`) ensures that we reuse the existing scope
    # named `scope`. Otherwise, TF creates a unique scope whose name starts with
    # `scope`.
    with graph.as_default(), graph.name_scope(scope + sep):
      with graph.name_scope(scope + sep + 'BatchNorm_Fold' + sep):
        # new weights = old weights * gamma / sqrt(variance + epsilon)
        # new biases = -mean * gamma / sqrt(variance + epsilon) + beta
        multiplier_tensor = match.gamma_tensor * math_ops.rsqrt(
            match.variance_tensor + match.bn_op.get_attr('epsilon'))
        bias_tensor = math_ops.subtract(
            match.beta_tensor,
            match.mean_tensor * multiplier_tensor,
            name='bias')

        correction_scale, correction_recip, correction_offset = None, None, None
        if is_training:
          correction_scale, correction_recip, correction_offset = (
              _ComputeBatchNormCorrections(
                  context='',
                  match=match,
                  freeze_batch_norm_delay=freeze_batch_norm_delay))
        # The shape of depthwise weights is different, so we need to reshape the
        # multiplier_tensor to ensure that the scaled_weight_tensor has the
        # expected shape.
        weights = match.weight_tensor

        # remember for the other loops
        matched_layer_set.add(match.layer_op)
        matched_layer_set.add(match.bn_op)

        if match.layer_op.type == 'DepthwiseConv2dNative':
          new_shape = [
              match.weight_tensor.get_shape().as_list()[2],
              match.weight_tensor.get_shape().as_list()[3]
          ]
          multiplier_tensor = array_ops.reshape(
              multiplier_tensor, new_shape, name='scale_reshape')

          if correction_scale is not None:
            correction_scale = array_ops.reshape(
                correction_scale, new_shape, name='correction_reshape')

        if correction_scale is not None:
          weights = math_ops.multiply(correction_scale, weights, name='correction_mult')

        scaled_weight_tensor = math_ops.multiply(
            weights, multiplier_tensor, name='mul_fold')

        # >>>>> CUSTOM >>>>>>>>>>>>>>
        # use hidden variable instead
        scaled_weight_tensor = variableFromSettings([],hiddenVar=scaled_weight_tensor)[0]
        # bias_tensor = variableFromSettings([],hiddenVar=bias_tensor)[0]
        # bias_tensor = next_base2(bias_tensor, strict_positive=False, min=1e-8)
        if S("util.variable.fixed_point.use"):
            bias_tensor = fixed_point(bias_tensor,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<

        new_layer_tensor = _CloneWithNewOperands(
            match.layer_op, match.input_tensor, scaled_weight_tensor,
            match.batch_to_space_op)

        if correction_recip is not None:
          new_layer_tensor = math_ops.multiply(
              correction_recip, new_layer_tensor, name='post_conv_mul')
          new_layer_tensor = math_ops.add(new_layer_tensor, (correction_offset), 'correction_add')
        if S("util.variable.fixed_point.use"):
            new_layer_tensor = fixed_point(new_layer_tensor,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))

        new_layer_tensor = math_ops.add( new_layer_tensor, bias_tensor, name='add_fold')
        if S("util.variable.fixed_point.use"):
            new_layer_tensor = tf.clip_by_value(new_layer_tensor,S("util.variable.fixed_point.min"),S("util.variable.fixed_point.max"))

        nodes_modified_count = common.RerouteTensor(new_layer_tensor, match.output_tensor)
        if nodes_modified_count == 0:
          raise ValueError('Folding batch norms failed, %s had no outputs.' % match.output_tensor.name)


def _FoldUnfusedBatchNorms(graph, is_training, freeze_batch_norm_delay):
  """Finds unfused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, True if training.
    freeze_batch_norm_delay: How many steps to wait before freezing moving mean
      and variance and using them for batch normalization.

  Raises:
    ValueError: When batch norm folding fails.
  """
  input_to_ops_map = input_to_ops.InputToOps(graph)

  for bn in common.BatchNormGroups(graph):
    has_scaling = _HasScaling(graph, input_to_ops_map, bn)

    if not _IsValidUnfusedBatchNorm(graph, bn):
      continue

    print("found unfused batchnarm")
    raise Exception("Not Implemented")

    # The mangling code intimately depends on BatchNorm node's internals.
    original_op, folded_op = _CreateFoldedOp(
        graph,
        bn,
        has_scaling=has_scaling,
        freeze_batch_norm_delay=freeze_batch_norm_delay,
        is_training=is_training)

    activation = common.GetEndpointActivationOp(graph, bn)
    if activation:
      nodes_modified_count = common.RerouteTensor(
          folded_op.outputs[0], original_op.outputs[0], can_modify=[activation])
      if nodes_modified_count != 1:
        raise ValueError('Unexpected inputs to op: %s' % activation.name)
      continue

    # Treat consumer ops in bypass modules differently since they have Add
    # operations instead of Relu* above.
    # Changes to make sure that the correct scope is selected for the bypass add
    # The rule here is that if the scope is of the form: str1/str2 for the
    # batch norm,
    # the bypass add is at scope str1. If bn is of scope just str1, then the
    # bypass add is at scope ''.
    # If there is no batch norm, then there is no bypass add.
    add_bypass_ctx = ''
    if bn:
      try:
        add_bypass_ctx = re.search(r'^(.*)/([^/]+)', bn).group(1)
      except AttributeError:
        add_bypass_ctx = ''

    if add_bypass_ctx:
      add_bypass_ctx = add_bypass_ctx + '/'

    add_bypass = graph.get_operation_by_name(add_bypass_ctx + 'Add')
    nodes_modified_count = common.RerouteTensor(
        folded_op.outputs[0], original_op.outputs[0], can_modify=[add_bypass])
    if nodes_modified_count != 1:
      raise ValueError('Unexpected inputs to op: %s' % add_bypass.name)

def _CloneWithNewOperands(layer_op, input_tensor, weight_tensor,
                          batch_to_space_op):
  """Clones layer_op with input_tensor and weight_tensor as new inputs."""
  new_layer_name = layer_op.name.split('/')[-1] + '_psb'
  if layer_op.type == 'Conv2D':
    return nn_ops.conv2d(
        input_tensor,
        weight_tensor,
        strides=layer_op.get_attr('strides'),
        padding=layer_op.get_attr('padding'),
        use_cudnn_on_gpu=layer_op.get_attr('use_cudnn_on_gpu'),
        data_format=layer_op.get_attr('data_format'),
        name=new_layer_name)
  elif layer_op.type == 'MatMul':
    return math_ops.matmul(
        input_tensor,
        weight_tensor,
        transpose_a=layer_op.get_attr('transpose_a'),
        transpose_b=layer_op.get_attr('transpose_b'),
        name=new_layer_name)
  elif layer_op.type == 'DepthwiseConv2dNative':
    # We don't copy dilation rate because we reuse the input SpaceToBatch
    # and create our own BatchToSpace operation below.
    conv = nn.depthwise_conv2d(
        input_tensor,
        weight_tensor,
        strides=layer_op.get_attr('strides'),
        padding=layer_op.get_attr('padding'),
        name=new_layer_name)
    # Copy the batch to space operation if we have a atrous convolution.
    if batch_to_space_op:
      batch_to_space_op = layer_op.outputs[0].consumers()[0]
      # Restructure this code to not rely on scope at all.
      new_batch_to_space_name = batch_to_space_op.name.split('/')[-1] + '_psb'
      conv = array_ops.batch_to_space_nd(
          conv,
          batch_to_space_op.inputs[1],
          batch_to_space_op.inputs[2],
          name=new_batch_to_space_name)
    return conv
  else:
    raise ValueError('Cannot handle operation of type: %s' % layer_op.type)


def _RedoRestFilters(graph):
  """Finds fused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, true if training.

  Raises:
    ValueError: When batch norm folding fails.
  """
  matches = _FindRestFilters(graph)
  print("Replacing",len(matches),"Conv|Mul|DepthwiseConv2dNative-Filters (without a suceeding BatchNorm)")
  for match in matches:
    scope, sep, _ = match['layer_op'].name.rpartition('/')
    # Make sure new ops are added to `graph` and put on the same device as
    # `bn_op`. The '/' (i.e. `sep`) ensures that we reuse the existing scope
    # named `scope`. Otherwise, TF creates a unique scope whose name starts with
    # `scope`.
    with graph.as_default(), graph.name_scope(scope + sep):
      with graph.name_scope(scope + sep + '_psb' + sep):

        weight = match['weight_tensor']

        # >>>>> CUSTOM >>>>>>>>>>>>>>
        # use hidden variable instead
        sampled_weight = variableFromSettings([],hiddenVar=weight)[0]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<

        new_layer_tensor = _CloneWithNewOperands(
          match['layer_op'], match['input_tensor'], sampled_weight,
          False)
        if S("util.variable.fixed_point.use"):
            new_layer_tensor = fixed_point(new_layer_tensor,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))

        nodes_modified_count = common.RerouteTensor(new_layer_tensor, match['output_tensor'])
        if nodes_modified_count == 0:
          raise ValueError('Folding batch norms failed, %s had no outputs.' % match['output_tensor'].name)


def _FindRestFilters(graph, find_unreplaced=True):
  """Finds all ops and tensors related to found FusedBatchNorms.

  Args:
    graph: Graph to inspect.

  Returns:
    _FusedBatchNormMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  weight_pattern = graph_matcher.OpTypePattern('*')
  layer_pattern = graph_matcher.OpTypePattern(
      'Conv2D|DepthwiseConv2dNative|MatMul',
      inputs=[input_pattern, weight_pattern])

  layer_pattern_matcher = graph_matcher.GraphMatcher(layer_pattern)

  def _GetLayerMatch(match_result):
    layer_op = match_result.get_op(layer_pattern)
    # layer_tensor = match_result.get_tensor(layer_pattern)
    input_tensor = match_result.get_tensor(input_pattern)
    weight_tensor = match_result.get_tensor(weight_pattern)
    output_tensor = layer_op.outputs[0]

    assert len(layer_op.outputs) == 1

    # Ensure that the output tensor has consumers, otherwise this is a dangling
    # node and not a match.
    if find_unreplaced and (not output_tensor.consumers() or layer_op.name.endswith("_psb")):
        return None, None

    return layer_op, {
        "layer_op": layer_op,
        "output_tensor": output_tensor,
        "input_tensor": input_tensor,
        "weight_tensor": weight_tensor
    }

  layer_matches = []
  for match_result in layer_pattern_matcher.match_graph(graph):
    layer_op, layer_match = _GetLayerMatch(match_result)
    if layer_op is not None:
      if layer_op not in matched_layer_set:
        matched_layer_set.add(layer_op)
        layer_matches.append(layer_match)

  return layer_matches














def _FindRestBatchNorms(graph):
  """Finds all ops and tensors related to found FusedBatchNorms.
  Args:
    graph: Graph to inspect.
  Returns:
    _FusedBatchNormMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  # In practice, the weight pattern can match a Variable or a SpaceToBatchND
  # operation that follows a variable for atrous convolutions.
  gamma_pattern = graph_matcher.OpTypePattern('*')
  beta_pattern = graph_matcher.OpTypePattern('*')
  mean_pattern = graph_matcher.OpTypePattern('*')
  variance_pattern = graph_matcher.OpTypePattern('*')

  moving_average_pattern = graph_matcher.OpTypePattern('*')
  bn_decay_pattern = graph_matcher.OpTypePattern('*')
  batch_to_space_pattern = graph_matcher.OpTypePattern(
      'BatchToSpaceND',
      inputs=[
          input_pattern,
          graph_matcher.OpTypePattern('*'),
          graph_matcher.OpTypePattern('*')
      ])

  batch_norm_pattern = graph_matcher.OpTypePattern(
      'FusedBatchNorm',
      inputs=[
          input_pattern, gamma_pattern,
          beta_pattern, mean_pattern, variance_pattern
      ])
  bn_matcher = graph_matcher.GraphMatcher(batch_norm_pattern)

  moving_average_sub_pattern = graph_matcher.OpTypePattern(
      'Sub', inputs=[moving_average_pattern, batch_norm_pattern])
  moving_average_mul_pattern = graph_matcher.OpTypePattern(
      'Mul', inputs=[moving_average_sub_pattern, bn_decay_pattern])

  moving_avg_mul_matcher = graph_matcher.GraphMatcher(
      moving_average_mul_pattern)

  def _GetLayerMatch(match_result):
    """Populates a layer match object containing ops/tensors for folding BNs.
    Args:
      match_result: Matched result from graph matcher
    Returns:
      BatchNormMatch: _BatchNormMatch containing all required batch norm
      parameters.
    """
    moving_mean_tensor = None
    moving_variance_tensor = None
    bn_decay_mean_tensor = None
    bn_decay_var_tensor = None
    batch_to_space_op = None
    bn_op = match_result.get_op(batch_norm_pattern)

    batch_epsilon = bn_op.get_attr('epsilon')

    # In the MatMul case, the output of batch norm is reshaped back into a
    # 2D tensor, so the output_tensor is the output of the Reshape op.
    output_tensor = bn_op.outputs[0]

    # Ensure that the output tensor has consumers, otherwise this is a dangling
    # node and not a match.
    if not output_tensor.consumers():
      return None, None

    batch_to_space_op = match_result.get_op(batch_to_space_pattern)
    input_tensor = match_result.get_tensor(input_pattern)
    gamma_tensor = match_result.get_tensor(gamma_pattern)
    beta_tensor = match_result.get_tensor(beta_pattern)
    # FusedBatchNorm in training is different from that in inference. It takes
    # empty 'mean' and empty 'variance', and produces the mean and the variance
    # of the batch. Therefore, when is_training is true, mean_tensor and
    # variance_tensor point to 1st and 2nd (0-based) output of bn_op,
    # respectively; when is_training is false, they point to bn_op's inputs.
    is_training = bn_op.get_attr('is_training')
    if is_training:
      # FusedBatchNormGrad doesn't compute gradients of the batch_mean and
      # batch_variance outputs, so we need to substitute our own custom
      # gradient.
      # pylint: disable=protected-access
      bn_op._set_attr(
          '_gradient_op_type',
          attr_value_pb2.AttrValue(s=compat.as_bytes('FoldFusedBatchNormGrad')))
      # pylint: enable=protected-access
      mean_tensor = bn_op.outputs[1]
      # The batch variance used during forward and backward prop is biased,
      # i.e it is calculated as: V=sum(x(k)-mu)^2/N. For the moving average
      # calculation, the variance is corrected by the term N/N-1 (Bessel's
      # correction). The variance tensor read from FuseBatchNorm has Bessel's
      # correction applied, so we undo it here.
      scope, sep, _ = bn_op.name.rpartition('/')
      g = ops.get_default_graph()
      with g.as_default(), g.name_scope(scope + sep):
        n = math_ops.cast(
            array_ops.size(input_tensor) / array_ops.size(mean_tensor),
            dtypes.float32)
        variance_tensor = math_ops.multiply(
            bn_op.outputs[2], (n - 1) / n, name='Undo_Bessel_Correction')
      for mul_match_result in moving_avg_mul_matcher.match_graph(graph):
        sub_op = mul_match_result.get_op(moving_average_sub_pattern)
        if sub_op.inputs[1].name == bn_op.outputs[1].name:
          # During training: Batch Mean is bn_op.outputs[1]
          moving_mean_tensor = sub_op.inputs[0]
          bn_decay_mean_tensor = mul_match_result.get_tensor(bn_decay_pattern)
        if sub_op.inputs[1].name == bn_op.outputs[2].name:
          # During training: Batch Var is bn_op.outputs[2]
          moving_variance_tensor = sub_op.inputs[0]
          bn_decay_var_tensor = mul_match_result.get_tensor(bn_decay_pattern)
    else:
      mean_tensor = match_result.get_tensor(mean_pattern)
      variance_tensor = match_result.get_tensor(variance_pattern)

    return bn_op, _BatchNormMatch(
        bn_op=bn_op,
        layer_op=None,
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        weight_tensor=None,
        gamma_tensor=gamma_tensor,
        beta_tensor=beta_tensor,
        mean_tensor=mean_tensor,
        variance_tensor=variance_tensor,
        moving_mean_tensor=moving_mean_tensor,
        moving_variance_tensor=moving_variance_tensor,
        bn_decay_mean_tensor=bn_decay_mean_tensor,
        bn_decay_var_tensor=bn_decay_var_tensor,
        batch_epsilon=batch_epsilon,
        batch_to_space_op=batch_to_space_op)

  layer_matches = []
  # We use matched_layer_set to ensure that layers aren't matched multiple
  # times.



  for match_result in bn_matcher.match_graph(graph):
    layer_op, layer_match = _GetLayerMatch(match_result)
    if layer_op is not None:
      if layer_op not in matched_layer_set:
        # print(layer_op.name, layer_match.output_tensor)
        matched_layer_set.add(layer_op)
        layer_matches.append(layer_match)

  # return []
  return layer_matches




def _RedoRestBatchnorms(graph, is_training):
  """Finds fused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, true if training.

  Raises:
    ValueError: When batch norm folding fails.
  """
  matches = _FindRestBatchNorms(graph)
  print("Replacing",len(matches),"BatchNorms (without a preceding Conv2D)")
  for match in matches:
    scope, sep, _ = match.bn_op.name.rpartition('/')
    # Make sure new ops are added to `graph` and put on the same device as
    # `bn_op`. The '/' (i.e. `sep`) ensures that we reuse the existing scope
    # named `scope`. Otherwise, TF creates a unique scope whose name starts with
    # `scope`.
    with graph.as_default(), graph.name_scope(scope + sep):
      with graph.name_scope(scope + sep + '_psb' + sep):

          mean = match.mean_tensor
          variance = match.variance_tensor
          beta = match.beta_tensor
          gamma = match.gamma_tensor
          eps = match.batch_epsilon

          # new gamma = gamma / sqrt(variance + epsilon)
          # new biases = -mean * gamma / sqrt(variance + epsilon) + beta
          multfac = gamma/math_ops.sqrt(variance+eps)
          gamma = multfac
          beta = -multfac*mean + beta
          mean = array_ops.zeros_like(mean)
          variance = array_ops.ones_like(variance)
          eps = array_ops.zeros_like(eps)

          gamma = variableFromSettings([],hiddenVar=gamma)[0]
          # gamma = fixed_point(gamma,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))
          # gamma = next_base2(gamma,strict_positive=False)
          # gamma = 1/variableFromSettings([],hiddenVar=1/gamma)[0]
          # variance = variableFromSettings([],hiddenVar=math_ops.sqrt(variance+eps))[0]**2
          # beta = variableFromSettings([],hiddenVar=beta)[0]
          if S("util.variable.fixed_point.use"):
              beta = fixed_point(beta,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))
              # gamma = fixed_point(gamma,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))
              # mean = fixed_point(mean,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))
              # variance = fixed_point(variance,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))

          # fixed_point division could be ok
          # silly silly_idiv(silly x, silly y) {
          #     uint64_t sign_bit = 1UL<<63;
          #     // unsetting the sign bit to ignore it
          #     silly res = ((x & ~sign_bit) / (y & sign_bit)) << 32;

          #     // setting the sign bit iff only one of sign bits is set
          #     res |= (x & sign_bit) ^ (y & sign_bit);
          #     return res;
          # }

      new_layer_tensor = nn.batch_normalization(match.input_tensor, mean, variance, beta, gamma, eps, name=match.bn_op.name.split("/")[-1]+"_psb")
      if S("util.variable.fixed_point.use"):
          new_layer_tensor = fixed_point(new_layer_tensor,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))
      nodes_modified_count = common.RerouteTensor(new_layer_tensor, match.output_tensor)
      if nodes_modified_count == 0:
        raise ValueError('Folding batch norms failed, %s had no outputs.' % match['output_tensor'].name)






def _RedoRestBias(graph):
  """Finds fused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, true if training.

  Raises:
    ValueError: When batch norm folding fails.
  """
  matches = _FindRestBias(graph)
  print("Replacing",len(matches),"BiasAdd")
  for match in matches:
    scope, sep, _ = match['layer_op'].name.rpartition('/')
    # Make sure new ops are added to `graph` and put on the same device as
    # `bn_op`. The '/' (i.e. `sep`) ensures that we reuse the existing scope
    # named `scope`. Otherwise, TF creates a unique scope whose name starts with
    # `scope`.
    with graph.as_default(), graph.name_scope(scope + sep):
      # with graph.name_scope(scope + sep + '_psb' + sep):

      bias = match['weight_tensor']

      # >>>>> CUSTOM >>>>>>>>>>>>>>
      # use hidden variable instead
      # bias = variableFromSettings([],hiddenVar=bias)[0]
      if S("util.variable.fixed_point.use"):
          bias = fixed_point(bias,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))
      # <<<<<<<<<<<<<<<<<<<<<<<<<<<

      new_layer_tensor = match['input_tensor'] + bias
      if S("util.variable.fixed_point.use"):
          new_layer_tensor = fixed_point(new_layer_tensor,S("util.variable.fixed_point.bits"),max=S("util.variable.fixed_point.max"),min=S("util.variable.fixed_point.min"))

      nodes_modified_count = common.RerouteTensor(new_layer_tensor, match['output_tensor'])
      if nodes_modified_count == 0:
        raise ValueError('Folding batch norms failed, %s had no outputs.' %
                         match['output_tensor'].name)


def _FindRestBias(graph):
  """Finds all ops and tensors related to found FusedBatchNorms.

  Args:
    graph: Graph to inspect.

  Returns:
    _FusedBatchNormMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  weight_pattern = graph_matcher.OpTypePattern('*')
  layer_pattern = graph_matcher.OpTypePattern(
      'BiasAdd',
      inputs=[input_pattern, weight_pattern])

  layer_pattern_matcher = graph_matcher.GraphMatcher(layer_pattern)

  def _GetLayerMatch(match_result):
    layer_op = match_result.get_op(layer_pattern)
    # layer_tensor = match_result.get_tensor(layer_pattern)
    input_tensor = match_result.get_tensor(input_pattern)
    weight_tensor = match_result.get_tensor(weight_pattern)
    output_tensor = layer_op.outputs[0]

    assert len(layer_op.outputs) == 1

    # Ensure that the output tensor has consumers, otherwise this is a dangling
    # node and not a match.
    if not output_tensor.consumers() or layer_op.name.endswith("_psb"):
        return None, None

    return layer_op, {
        "layer_op": layer_op,
        "output_tensor": output_tensor,
        "input_tensor": input_tensor,
        "weight_tensor": weight_tensor
    }

  layer_matches = []
  for match_result in layer_pattern_matcher.match_graph(graph):
    layer_op, layer_match = _GetLayerMatch(match_result)
    if layer_op is not None:
      if layer_op not in matched_layer_set:
        matched_layer_set.add(layer_op)
        layer_matches.append(layer_match)

  return layer_matches




def _RedoRestAvgPool(graph):
  """Finds fused batch norm layers and folds them into preceding layers.

  Folding only affects the following layers: Conv2D, fully connected, depthwise
  convolution.

  Args:
    graph: Graph to walk and modify.
    is_training: Bool, true if training.

  Raises:
    ValueError: When batch norm folding fails.
  """
  matches = _FindRestAvgPool(graph)
  print("Replacing",len(matches),"AvgPool")
  for match in matches:
    scope, sep, _ = match['layer_op'].name.rpartition('/')
    # Make sure new ops are added to `graph` and put on the same device as
    # `bn_op`. The '/' (i.e. `sep`) ensures that we reuse the existing scope
    # named `scope`. Otherwise, TF creates a unique scope whose name starts with
    # `scope`.
    with graph.as_default(), graph.name_scope(scope + sep):
      # with graph.name_scope(scope + sep + '_psb' + sep):

      input_tensor = match['input_tensor']
      layer_op = match['layer_op']
      # output_tensor = match['output_tensor']

      # >>>>> CUSTOM >>>>>>>>>>>>>>
      avg_size = np.prod(layer_op.get_attr("ksize")).astype(np.float32)
      if avg_size == 2**np.log2(avg_size):
          continue
      output_tensor = nn_ops.avg_pool(
              input_tensor,
              ksize=layer_op.get_attr('ksize'),
              strides=layer_op.get_attr('strides'),
              padding=layer_op.get_attr('padding'),
              data_format=layer_op.get_attr('data_format'),
              name=layer_op.name.split('/')[-1] + '_psb'
      )
      avg_size_new = variableFromSettings([],hiddenVar=(1.0/avg_size).astype(np.float32))[0]
      new_layer_tensor = output_tensor*avg_size*avg_size_new
      # <<<<<<<<<<<<<<<<<<<<<<<<<<<

      nodes_modified_count = common.RerouteTensor(new_layer_tensor, match['output_tensor'])
      if nodes_modified_count == 0:
        raise ValueError('Folding batch norms failed, %s had no outputs.' % match['output_tensor'].name)


def _FindRestAvgPool(graph):
  """Finds all ops and tensors related to found FusedBatchNorms.

  Args:
    graph: Graph to inspect.

  Returns:
    _FusedBatchNormMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  layer_pattern = graph_matcher.OpTypePattern(
      'AvgPool',
      inputs=[input_pattern])

  layer_pattern_matcher = graph_matcher.GraphMatcher(layer_pattern)

  def _GetLayerMatch(match_result):
    layer_op = match_result.get_op(layer_pattern)
    # layer_tensor = match_result.get_tensor(layer_pattern)
    input_tensor = match_result.get_tensor(input_pattern)
    output_tensor = layer_op.outputs[0]

    assert len(layer_op.outputs) == 1

    # Ensure that the output tensor has consumers, otherwise this is a dangling
    # node and not a match.
    if not output_tensor.consumers() or layer_op.name.endswith("_psb"):
        return None, None

    return layer_op, {
        "layer_op": layer_op,
        "output_tensor": output_tensor,
        "input_tensor": input_tensor,
    }

  layer_matches = []
  for match_result in layer_pattern_matcher.match_graph(graph):
    layer_op, layer_match = _GetLayerMatch(match_result)
    if layer_op is not None:
      if layer_op not in matched_layer_set:
        matched_layer_set.add(layer_op)
        layer_matches.append(layer_match)

  return layer_matches





