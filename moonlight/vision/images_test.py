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
"""Tests for image utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from moonlight.vision import images


class ImagesTest(tf.test.TestCase):

  def testTranslate(self):
    with self.test_session():
      arr = tf.reshape(tf.range(9), (3, 3))
      self.assertAllEqual(
          images.translate(arr, 0, -1).eval(),
          [[3, 4, 5], [6, 7, 8], [0, 0, 0]])
      self.assertAllEqual(
          images.translate(arr, 0, 1).eval(), [[0, 0, 0], [0, 1, 2], [3, 4, 5]])
      self.assertAllEqual(
          images.translate(arr, -1, 0).eval(),
          [[1, 2, 0], [4, 5, 0], [7, 8, 0]])
      self.assertAllEqual(
          images.translate(arr, 1, 0).eval(), [[0, 0, 1], [0, 3, 4], [0, 6, 7]])


if __name__ == '__main__':
  tf.test.main()
