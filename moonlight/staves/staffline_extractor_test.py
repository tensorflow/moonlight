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

import os.path

import numpy as np
import tensorflow as tf

from moonlight import staves
from moonlight.staves import staffline_extractor


class StafflineExtractorTest(tf.test.TestCase):

  def setUp(self):
    # Small image with a single staff.
    # pyformat: disable
    self.single_staff_image = np.asarray(
        [[1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1],
         [1, 0, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1]], np.uint8) * 255

  def testExtractStaff(self):
    image_t = tf.constant(self.single_staff_image, name='image')
    detector = staves.ProjectionStaffDetector(image_t)
    # The staffline distance is 3, so use a target height of 6 to avoid scaling
    # the image.
    extractor = staffline_extractor.StafflineExtractor(
        image_t,
        detector,
        target_height=6,
        num_sections=9,
        staffline_distance_multiple=2)
    with self.test_session():
      stafflines = extractor.extract_staves().eval()
    assert stafflines.shape == (1, 9, 6, 7)
    # The top staff line is at a y-value of 2 because of rounding.
    assert np.array_equal(
        stafflines[0, 0],
        np.concatenate((np.zeros((2, 7)), self.single_staff_image[:4] / 255.0)))
    # The staff space is centered in the window.
    assert np.array_equal(stafflines[0, 3],
                          self.single_staff_image[3:9] / 255.0)
    # Staffline height is 3 and extracted strips have 2 staff line distances
    # with a total height of 6, so the strip is not actually scaled.
    self.assertTrue(
        np.logical_or(np.isclose(stafflines, 1.), np.isclose(stafflines,
                                                             0.)).all())

  def testFloatMultiple(self):
    image_t = tf.constant(self.single_staff_image, name='image')
    detector = staves.ProjectionStaffDetector(image_t)
    extractor = staffline_extractor.StafflineExtractor(
        image_t,
        detector,
        target_height=6,
        num_sections=9,
        staffline_distance_multiple=1.5)
    with self.test_session():
      stafflines = extractor.extract_staves().eval()
    # Staff strip is scaled up because the input height is less. Some output
    # pixels have aliasing.
    self.assertAllEqual(stafflines.shape, (1, 9, 6, 10))
    self.assertFalse(
        np.logical_or(np.isclose(stafflines, 1.), np.isclose(stafflines,
                                                             0.)).all())


class StafflinePatchExtractorTest(tf.test.TestCase):

  def testCompareIteratorWithSinglePatch(self):
    # Patches from the iterator should be exactly equal to the patch when the
    # coordinates from the id are given.
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../testdata/IMSLP00747-000.png')

    extractor = staffline_extractor.StafflinePatchExtractor()
    with self.test_session(graph=extractor.graph):
      single_patch = extractor.extract_staff_patch(filename, 3, 4, 800)
      # Unwrap the single patch with the matching id in the iterator.
      patch_from_iterator, = (
          patch
          for patch_id, patch in extractor.page_patch_iterator(filename)
          if patch_id == 'IMSLP00747-000,3,+4,800'
      )
    self.assertAllClose(single_patch, patch_from_iterator)
    # Patch contains some content.
    self.assertAlmostEqual(single_patch.min(), 0, places=5)
    self.assertAlmostEqual(single_patch.max(), 1, places=5)


if __name__ == '__main__':
  tf.test.main()
