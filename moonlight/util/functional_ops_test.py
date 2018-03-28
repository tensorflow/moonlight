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
"""Tests for the functional ops helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from moonlight.util import functional_ops


class FunctionalOpsTest(tf.test.TestCase):

  def testFlatMap(self):
    with self.test_session():
      items = functional_ops.flat_map_fn(tf.range, [1, 3, 0, 5])
      self.assertAllEqual(items.eval(), [0, 0, 1, 2, 0, 1, 2, 3, 4])


if __name__ == '__main__':
  tf.test.main()
