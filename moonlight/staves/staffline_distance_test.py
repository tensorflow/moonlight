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
"""Tests for staffline distance estimation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

from moonlight.image import decode_music_score_png
from moonlight.staves import staffline_distance


class StafflineDistanceTest(tf.test.TestCase):

  def testCorpusImage(self):
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../testdata/IMSLP00747-000.png')
    image_contents = open(filename, 'rb').read()
    image_t = decode_music_score_png(tf.constant(image_contents))
    staffdist_t, staffthick_t = (
        staffline_distance.estimate_staffline_distance_and_thickness(image_t,))
    with self.test_session() as sess:
      staffdist, staffthick = sess.run((staffdist_t, staffthick_t))
    # Manually determined values for the image.
    self.assertAllEqual(staffdist, [16])
    self.assertEquals(staffthick, 2)

  def testZeros(self):
    # All white (0) shouldn't be picked up as a music score.
    image_t = tf.zeros((512, 512), dtype=tf.uint8)
    staffdist_t, staffthick_t = (
        staffline_distance.estimate_staffline_distance_and_thickness(image_t))
    with self.test_session() as sess:
      staffdist, staffthick = sess.run((staffdist_t, staffthick_t))
    self.assertAllEqual(staffdist, [])
    self.assertEqual(staffthick, -1)

  def testSpeckles(self):
    # Random speckles shouldn't be picked up as a music score.
    tf.set_random_seed(1234)
    image_t = tf.where(
        tf.random_uniform((512, 512)) < 0.1,
        tf.fill((512, 512), tf.constant(255, tf.uint8)),
        tf.fill((512, 512), tf.constant(0, tf.uint8)))
    staffdist_t, staffthick_t = (
        staffline_distance.estimate_staffline_distance_and_thickness(image_t))
    with self.test_session() as sess:
      staffdist, staffthick = sess.run((staffdist_t, staffthick_t))
    self.assertAllEqual(staffdist, [])
    self.assertEqual(staffthick, -1)


if __name__ == '__main__':
  tf.test.main()
