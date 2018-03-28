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
"""Tests for binary morphology."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from moonlight.vision import morphology


class MorphologyTest(tf.test.TestCase):

  def testMorphology_false(self):
    for op in [morphology.binary_erosion, morphology.binary_dilation]:
      with self.test_session():
        self.assertAllEqual(
            op(tf.zeros((5, 3), tf.bool), n=1).eval(), np.zeros((5, 3),
                                                                np.bool))

  def testErosion_small(self):
    with self.test_session():
      self.assertAllEqual(
          morphology.binary_erosion(
              tf.cast([[0, 1, 0], [1, 1, 1], [0, 1, 0]], tf.bool), n=1).eval(),
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]])

  def testErosion(self):
    with self.test_session():
      self.assertAllEqual(
          morphology.binary_erosion(
              tf.cast(
                  [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]],
                  tf.bool),
              n=1).eval(),
          np.asarray(
              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
              np.bool))  # pyformat: disable

  def testDilation(self):
    with self.test_session():
      self.assertAllEqual(
          morphology.binary_dilation(
              tf.cast(
                  [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]],
                  tf.bool),
              n=1).eval(),
          np.asarray(
              [[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]],
              np.bool))  # pyformat: disable


if __name__ == '__main__':
  tf.test.main()
