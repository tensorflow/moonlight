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
"""Tests for structure computation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tensorflow as tf

from moonlight import image as image_module
from moonlight import structure


class StructureTest(tf.test.TestCase):

  def testCompute(self):
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../testdata/IMSLP00747-000.png')
    image = image_module.decode_music_score_png(tf.read_file(filename))
    struct = structure.create_structure(image)
    with self.test_session():
      struct = struct.compute()
    self.assertEqual(np.int32, struct.staff_detector.staves.dtype)
    # Expected number of staves for the corpus image.
    self.assertEqual((12, 2, 2), struct.staff_detector.staves.shape)

    self.assertEqual(np.int32, struct.verticals.lines.dtype)
    self.assertEqual(3, struct.verticals.lines.ndim)
    self.assertEqual((2, 2), struct.verticals.lines.shape[1:])


if __name__ == '__main__':
  tf.test.main()
