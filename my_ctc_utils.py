import numpy as np
import keras
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_module
#from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_module
from keras.models import Sequential
from keras.layers import Conv1D,Conv2D,Dense, Dropout,Flatten,Bidirectional, Activation,BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import SGD
from keras.models import Sequential,Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.backend import ctc_decode, variable,get_value,concatenate
from keras.backend import *
import keras.backend as K
from keras import callbacks,losses
from keras.layers import Dense, Activation,Input,LSTM, Lambda
def my_ctc_decode(y_pred, input_length, greedy=False, beam_width=100, top_paths=1):
  """Decodes the output of a softmax.
  Can use either greedy search (also known as best path)
  or a constrained dictionary search.
  Arguments:
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, )` containing the sequence length for
          each batch item in `y_pred`.
      greedy: perform much faster best-path search if `true`.
          This does not use a dictionary.
      beam_width: if `greedy` is `false`: a beam search decoder will be used
          with a beam of this width.
      top_paths: if `greedy` is `false`,
          how many of the most probable paths will be returned.
  Returns:
      Tuple:
          List: if `greedy` is `true`, returns a list of one element that
              contains the decoded sequence.
              If `false`, returns the `top_paths` most probable
              decoded sequences.
              Important: blank labels are returned as `-1`.
          Tensor `(top_paths, )` that contains
              the log probability of each decoded sequence.
  """
  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
  input_length = math_ops.to_int32(input_length)

  if greedy:
    (decoded, log_prob) = ctc.ctc_greedy_decoder(
        inputs=y_pred, sequence_length=input_length)
  else:
    (decoded, log_prob) = my_ctc_beam_search_decoder(
        inputs=y_pred,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=top_paths)
  decoded_dense = [
      sparse_ops.sparse_to_dense(
          st.indices, st.dense_shape, st.values, default_value=-1)
      for st in decoded
  ]
  return (decoded_dense, log_prob)

def my_ctc_beam_search_decoder(inputs, sequence_length, beam_width=100,
                            top_paths=1, merge_repeated=False):
  """Performs beam search decoding on the logits given in input.
  **Note** The `ctc_greedy_decoder` is a special case of the
  `ctc_beam_search_decoder` with `top_paths=1` and `beam_width=1` (but
  that decoder is faster for this special case).
  If `merge_repeated` is `True`, merge repeated classes in the output beams.
  This means that if consecutive entries in a beam are the same,
  only the first of these is emitted.  That is, when the sequence is
  `A B B * B * B` (where '*' is the blank label), the return value is:
    * `A B` if `merge_repeated = True`.
    * `A B B B` if `merge_repeated = False`.
  Args:
    inputs: 3-D `float` `Tensor`, size
      `[max_time x batch_size x num_classes]`.  The logits.
    sequence_length: 1-D `int32` vector containing sequence lengths,
      having size `[batch_size]`.
    beam_width: An int scalar >= 0 (beam search beam width).
    top_paths: An int scalar >= 0, <= beam_width (controls output size).
    merge_repeated: Boolean.  Default: True.
  Returns:
    A tuple `(decoded, log_probabilities)` where
    decoded: A list of length top_paths, where `decoded[j]`
      is a `SparseTensor` containing the decoded outputs:
      `decoded[j].indices`: Indices matrix `(total_decoded_outputs[j] x 2)`
        The rows store: [batch, time].
      `decoded[j].values`: Values vector, size `(total_decoded_outputs[j])`.
        The vector stores the decoded classes for beam j.
      `decoded[j].dense_shape`: Shape vector, size `(2)`.
        The shape values are: `[batch_size, max_decoded_length[j]]`.
    log_probability: A `float` matrix `(batch_size x top_paths)` containing
        sequence log-probabilities.
  """

  decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
      gen_ctc_ops.ctc_beam_search_decoder(
          inputs, sequence_length, beam_width=beam_width, top_paths=top_paths,
          merge_repeated=merge_repeated))

  return (
      [sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
       in zip(decoded_ixs, decoded_vals, decoded_shapes)],
      log_probabilities)

#@tf_export('keras.backend.ctc_label_dense_to_sparse')
def my_ctc_label_dense_to_sparse(labels, label_lengths):
  """Converts CTC labels from dense to sparse.
  Arguments:
      labels: dense CTC labels.
      label_lengths: length of the labels.
  Returns:
      A sparse tensor representation of the labels.
  """
  label_shape = array_ops.shape(labels)
  num_batches_tns = array_ops.stack([label_shape[0]])
  max_num_labels_tns = array_ops.stack([label_shape[1]])

  def range_less_than(_, current_input):
    return array_ops.expand_dims(
        math_ops.range(label_shape[1]), 0) < array_ops.fill(
            max_num_labels_tns, current_input)

  init = math_ops.cast(
      array_ops.fill([1, label_shape[1]], 0), dtypes_module.bool)
  dense_mask = functional_ops.scan(
      range_less_than, label_lengths, initializer=init, parallel_iterations=1)
  dense_mask = dense_mask[:, 0, :]

  label_array = array_ops.reshape(
      array_ops.tile(math_ops.range(0, label_shape[1]), num_batches_tns),
      label_shape)
  label_ind = array_ops.boolean_mask(label_array, dense_mask)

  batch_array = array_ops.transpose(
      array_ops.reshape(
          array_ops.tile(math_ops.range(0, label_shape[0]), max_num_labels_tns),
          reverse(label_shape, 0)))
  batch_ind = array_ops.boolean_mask(batch_array, dense_mask)
  indices = array_ops.transpose(
      array_ops.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1]))

  vals_sparse = array_ops.gather_nd(labels, indices)

  return sparse_tensor.SparseTensor(
      math_ops.to_int64(indices), vals_sparse, math_ops.to_int64(label_shape))

def my_ctc_batch_cost(y_true, y_pred, input_length, label_length):
  """Runs CTC loss algorithm on each batch element.
  Arguments:
      y_true: tensor `(samples, max_string_length)`
          containing the truth labels.
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_pred`.
      label_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_true`.
  Returns:
      Tensor with shape (samples,1) containing the
          CTC loss of each element.
  """
  label_length = math_ops.to_int32(array_ops.squeeze(label_length, axis=-1))
  input_length = math_ops.to_int32(array_ops.squeeze(input_length, axis=-1))
  sparse_labels = math_ops.to_int32(
      my_ctc_label_dense_to_sparse(y_true, label_length))

  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())

  return array_ops.expand_dims(
      ctc.ctc_loss(
          inputs=y_pred, labels=sparse_labels, sequence_length=input_length,ctc_merge_repeated=False), 1)
