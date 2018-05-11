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
"""Defines the glyph patches DNN classifier. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from moonlight.models.base import glyph_patches
from moonlight.protobuf import musicscore_pb2
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_multi_integer(
    'layer_dims', [20, 20], 'Dimensions of each hidden layer.')
flags.DEFINE_float('learning_rate', 0.1, 'FTRL learning rate')
flags.DEFINE_float('l1_regularization_strength', 0.01, 'L1 penalty')
flags.DEFINE_float('l2_regularization_strength', 0, 'L2 penalty')


def create_estimator():
  return tf.estimator.DNNClassifier(
      FLAGS.layer_dims,
      feature_columns=[glyph_patches.create_patch_feature_column()],
      n_classes=len(musicscore_pb2.Glyph.Type.keys()),
      optimizer=tf.train.FtrlOptimizer(
          learning_rate=FLAGS.learning_rate,
          l1_regularization_strength=FLAGS.l1_regularization_strength,
          l2_regularization_strength=FLAGS.l2_regularization_strength,
      ),
      model_dir=glyph_patches.FLAGS.model_dir,
  )
