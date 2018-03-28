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
"""Tests for connected component analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from moonlight.structure import components


class ComponentsTest(tf.test.TestCase):

  def testComponents(self):
    arr = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
           [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
           [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]  # pyformat: disable
    component_bounds_t = components.get_component_bounds(tf.cast(arr, tf.bool))
    with self.test_session():
      component_bounds = component_bounds_t.eval()
    self.assertAllEqual(
        component_bounds,
        # x0, y0, x1, y1, size
        [[5, 0, 5, 0, 1], [0, 1, 4, 5, 16], [8, 1, 10, 3, 5], [12, 1, 12, 2, 2],
         [2, 3, 2, 3, 1], [6, 4, 6, 4, 1]])


if __name__ == '__main__':
  tf.test.main()
