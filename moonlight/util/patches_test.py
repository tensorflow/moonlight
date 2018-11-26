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
"""Tests for the patches utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from moonlight.util import patches
from six import moves


class PatchesTest(tf.test.TestCase):

  def test2D(self):
    image_t = tf.random_uniform((100, 200))
    image_t.set_shape((100, 200))
    patch_width = 10
    patches_t = patches.patches_1d(image_t, patch_width)
    with self.test_session() as sess:
      image_arr, patches_arr = sess.run((image_t, patches_t))
      self.assertEqual(patches_arr.shape,
                       (200 - patch_width + 1, 100, patch_width))
      for i in moves.xrange(patches_arr.shape[0]):
        self.assertAllEqual(patches_arr[i], image_arr[:, i:i + patch_width])

  def test4D(self):
    height = 15
    width = 20
    image_t = tf.random_uniform((4, 8, height, width))
    image_t.set_shape((None, None, height, width))
    patch_width = 10
    patches_t = patches.patches_1d(image_t, patch_width)
    with self.test_session() as sess:
      image_arr, patches_arr = sess.run((image_t, patches_t))
      self.assertEqual(patches_arr.shape,
                       (4, 8, width - patch_width + 1, height, patch_width))
      for i in moves.xrange(patches_arr.shape[0]):
        for j in moves.xrange(patches_arr.shape[1]):
          for k in moves.xrange(patches_arr.shape[2]):
            self.assertAllEqual(patches_arr[i, j, k],
                                image_arr[i, j, :, k:k + patch_width])

  def testEmpty(self):
    height = 15
    width = 20
    image_t = tf.zeros((1, 2, 0, 3, 4, height, width))
    patch_width = 10
    patches_t = patches.patches_1d(image_t, patch_width)
    with self.test_session():
      self.assertEqual(patches_t.eval().shape,
                       (1, 2, 0, 3, 4, 0, height, patch_width))


if __name__ == '__main__':
  tf.test.main()
