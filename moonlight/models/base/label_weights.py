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
"""Configures the weight (importance) of examples with a given label.

Any glyph type may be over- or under-represented in the examples, which would
hurt the precision and/or recall for that glyph type. When training, the
gradient for each example is multiplied by the weight, which scales the
parameter update for that example.

For an example custom weight, if naturals are often misclassified as sharps, and
not vice versa, we may want to increase the weight for NATURAL.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from moonlight.protobuf import musicscore_pb2
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'label_weights', 'NONE=0.5',
    'Example weights for patches of each label type. For example,'
    ' "NONE=0.01,FLAT=2.0" would weight "NONE" examples\' influence as 0.01,'
    ' "FLAT" examples as 2.0, and all other examples as 1.0.')

# The glyph types array must be large enough to hold the highest enum value.
GLYPH_TYPES_ARRAY_SIZE = 1 + max(
    number for name, number in musicscore_pb2.Glyph.Type.items())


def parse_label_weights_array(weights_str=None):
  """Creates an array with all of the label weights.

  Args:
    weights_str: String of label name-weight pairs, separated by commas.
        Defaults to the command-line flag.

  Returns:
    A NumPy array large enough to hold all of the glyph enum types. At the index
    for a glyph enum value, we store the example weight, defaulting to 1.0.

  Raises:
    ValueError: If a glyph type is listed multiple times.
  """
  weights_str = weights_str or FLAGS.label_weights
  weights_array = np.ones(GLYPH_TYPES_ARRAY_SIZE)
  if not weights_str:
    return weights_array

  weights = {}
  for pair in weights_str.split(','):
    name, glyph_weight_str = pair.split('=')
    if name in weights:
      raise ValueError('Duplicate weight: {}'.format(name))
    weights[name] = float(glyph_weight_str)

  for name, weight in weights.iteritems():
    weights_array[musicscore_pb2.Glyph.Type.Value(name)] = weight
  return weights_array


def weights_from_labels(labels, weights_str=None):
  """Determines the example weights from a tensor of example labels."""
  with tf.name_scope('weights_from_labels'):
    weights = tf.constant(
        parse_label_weights_array(weights_str), name='label_weights')
    return tf.gather(weights, labels, name='label_weights_lookup')
