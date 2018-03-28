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
"""Tests for run length encoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from moonlight.util import run_length


class RunLengthTest(tf.test.TestCase):

  def testEmpty(self):
    with self.test_session() as sess:
      columns, values, lengths = sess.run(
          run_length.vertical_run_length_encoding(tf.zeros((0, 0), tf.bool)))
    self.assertAllEqual(columns, [])
    self.assertAllEqual(values, [])
    self.assertAllEqual(lengths, [])

  def testBooleanImage(self):
    img = tf.cast(
        [
            [0, 0, 1, 0, 0, 1],
            # pyformat: disable
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 1, 0]
        ],
        tf.bool)
    with self.test_session() as sess:
      columns, values, lengths = sess.run(
          run_length.vertical_run_length_encoding(img))
    self.assertAllEqual(columns,
                        [0] * 3 + [1] * 4 + [2] + [3] * 3 + [4] * 2 + [5] * 2)
    self.assertAllEqual(values, [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    self.assertAllEqual(lengths, [1, 1, 2, 1, 1, 1, 1, 4, 1, 2, 1, 1, 3, 3, 1])


if __name__ == '__main__':
  tf.test.main()
