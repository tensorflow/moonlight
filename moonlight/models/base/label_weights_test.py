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
"""Tests for label_weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from moonlight.models.base import label_weights
from moonlight.protobuf import musicscore_pb2
import tensorflow as tf


class LabelWeightsTest(tf.test.TestCase):

  def testWeightsFromLabels(self):
    g = musicscore_pb2.Glyph
    labels = tf.constant(
        [g.NONE, g.NONE, g.NOTEHEAD_FILLED, g.SHARP, g.FLAT, g.NATURAL])
    weights = 'NONE=0.1,NATURAL=2.0,SHARP=0.5,NOTEHEAD_FILLED=0.8'
    weights_tensor = label_weights.weights_from_labels(labels, weights)
    with self.test_session():
      self.assertAllEqual([0.1, 0.1, 0.8, 0.5, 1.0, 2.0], weights_tensor.eval())


if __name__ == '__main__':
  tf.test.main()
