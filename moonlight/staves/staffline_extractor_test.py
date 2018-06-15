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
"""Tests for StafflineExtractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from moonlight import staves
from moonlight.staves import staffline_extractor


class StafflineExtractorTest(tf.test.TestCase):

  def testExtractStaff(self):
    # Small image with a single staff.
    image = np.asarray([[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]],
                       np.uint8) * 255
    image_t = tf.constant(image, name='image')
    detector = staves.ProjectionStaffDetector(image_t)
    # The staffline distance is 3, so use a target height of 6 to avoid scaling
    # the image.
    extractor = staffline_extractor.StafflineExtractor(
        image_t, detector, target_height=6, num_sections=9,
        staffline_distance_multiple=2)
    with self.test_session():
      stafflines = extractor.extract_staves().eval()
    assert stafflines.shape == (1, 9, 6, 7)
    # The top staff line is at a y-value of 2 because of rounding.
    assert np.array_equal(stafflines[0, 0],
                          np.concatenate((np.zeros((2, 7)), image[:4] / 255.0)))
    # The staff space is centered in the window.
    assert np.array_equal(stafflines[0, 3], image[3:9] / 255.0)


class StafflinePatchExtractorTest(tf.test.TestCase):

  def testCreateExtractor(self):
    """Ensures that we can initialize the extractor without errors."""
    # TODO(ringwalt): Test a real image.
    staffline_extractor.StafflinePatchExtractor()


if __name__ == '__main__':
  tf.test.main()
