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
"""Tests for staff removal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tensorflow as tf

from moonlight import image as omr_image
from moonlight import staves
from moonlight.staves import removal
from moonlight.staves import staffline_distance


class RemovalTest(tf.test.TestCase):

  def test_corpus_image(self):
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../testdata/IMSLP00747-000.png')
    image_t = omr_image.decode_music_score_png(tf.read_file(filename))
    remover = removal.StaffRemover(staves.StaffDetector(image_t))
    with self.test_session() as sess:
      removed, image = sess.run([remover.remove_staves, image_t])
      self.assertFalse(np.allclose(removed, image))
      # If staff removal runs successfully, we should be unable to estimate the
      # staffline distance from the staves-removed image.
      est_staffline_distance, est_staffline_thickness = sess.run(
          staffline_distance.estimate_staffline_distance_and_thickness(removed))
      print(est_staffline_distance)
      self.assertAllEqual([], est_staffline_distance)
      self.assertEqual(-1, est_staffline_thickness)


if __name__ == '__main__':
  tf.test.main()
