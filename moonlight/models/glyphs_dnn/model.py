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
"""Defines the glyph patches DNN classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from moonlight.models.base import glyph_patches
from moonlight.protobuf import musicscore_pb2
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_multi_integer(
    'layer_dims', [20, 20],
    'Dimensions of each hidden layer. --layer_dims=0 indicates logistic'
    ' regression (predictions directly connected to inputs through a sigmoid'
    ' layer).')
flags.DEFINE_string(
    'activation_fn', 'sigmoid',
    'The name of the function (under tf.nn) to apply after each layer.')
flags.DEFINE_float('learning_rate', 0.1, 'FTRL learning rate')
flags.DEFINE_float('l1_regularization_strength', 0.01, 'L1 penalty')
flags.DEFINE_float('l2_regularization_strength', 0, 'L2 penalty')
flags.DEFINE_float('dropout', None, 'Dropout to apply to all hidden nodes.')


def create_estimator():
  """Returns the DNNClassifier estimator based on the command-line flags.

  Returns:
    A DNNClassifier instance.
  """
  layer_dims = FLAGS.layer_dims
  if not any(layer_dims):
    # Must pass a single layer of size 0 on the command line to indicate
    # logistic regression (no hidden dims).
    layer_dims = []
  if not layer_dims and FLAGS.activation_fn != 'sigmoid':
    tf.logging.warning(
        'activation_fn should be sigmoid for logistic regression. Got: %s',
        FLAGS.activation_fn)

  activation_fn = getattr(tf.nn, FLAGS.activation_fn)
  return tf.estimator.DNNClassifier(
      layer_dims,
      feature_columns=[glyph_patches.create_patch_feature_column()],
      weight_column=glyph_patches.WEIGHT_COLUMN_NAME,
      n_classes=len(musicscore_pb2.Glyph.Type.keys()),
      optimizer=tf.train.FtrlOptimizer(
          learning_rate=FLAGS.learning_rate,
          l1_regularization_strength=FLAGS.l1_regularization_strength,
          l2_regularization_strength=FLAGS.l2_regularization_strength,
      ),
      activation_fn=activation_fn,
      dropout=FLAGS.dropout,
      model_dir=glyph_patches.FLAGS.model_dir,
  )
