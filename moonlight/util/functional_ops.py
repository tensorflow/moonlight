# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functional op helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def flat_map_fn(fn, elems, dtype=None):
  """Flat maps `fn` on items unpacked from `elems` on dimension 0.

  Analogous to `tf.map_fn`, but concatenates the result(s) along dimension 0.

  Args:
    fn: The callable to be performed.
    elems: The tensor of elements to apply `fn` to.
    dtype: The dtype of the output of `fn`.

  Returns:
    A tensor with the same rank as the input, and same dimensions except for
        dimension 0. The function results for each element, concatenated along
        dimension 0.
  """
  elems = tf.convert_to_tensor(elems)
  n = tf.shape(elems)[0]

  zero_elem = tf.zeros(tf.shape(elems)[1:], elems.dtype)
  dummy_output = fn(zero_elem)
  output_elem_shape = tf.shape(dummy_output)[1:]

  initial_results = tf.zeros(
      tf.concat([[0], output_elem_shape], axis=0), dtype=dtype or elems.dtype)

  def compute(i, results):
    elem_results = fn(elems[i])
    return i + 1, tf.concat([results, elem_results], axis=0)

  return tf.while_loop(
      lambda i, _: i < n,
      compute, [0, initial_results],
      shape_invariants=[tf.TensorShape(()),
                        tf.TensorShape(None)],
      parallel_iterations=1)[1]
