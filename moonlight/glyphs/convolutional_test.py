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
"""Tests for Convolutional1DGlyphClassifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from moonlight.glyphs import base
from moonlight.glyphs import convolutional
from moonlight.glyphs import testing
from moonlight.protobuf import musicscore_pb2

STAFF_INDEX = base.GlyphsTensorColumns.STAFF_INDEX
Y_POSITION = base.GlyphsTensorColumns.Y_POSITION
X = base.GlyphsTensorColumns.X
TYPE = base.GlyphsTensorColumns.TYPE


class ConvolutionalTest(tf.test.TestCase):

  def testGetGlyphsPage(self):
    # Refer to testing.py for the glyphs array.
    # pyformat: disable
    glyphs = pd.DataFrame(
        [
            {STAFF_INDEX: 0, Y_POSITION: 0, X: 0, TYPE: 3},
            {STAFF_INDEX: 0, Y_POSITION: -1, X: 1, TYPE: 4},
            {STAFF_INDEX: 0, Y_POSITION: 0, X: 2, TYPE: 5},
            {STAFF_INDEX: 0, Y_POSITION: 1, X: 4, TYPE: 2},
            {STAFF_INDEX: 1, Y_POSITION: 1, X: 2, TYPE: 3},
            {STAFF_INDEX: 1, Y_POSITION: 0, X: 2, TYPE: 5},
            {STAFF_INDEX: 1, Y_POSITION: -1, X: 4, TYPE: 3},
            {STAFF_INDEX: 1, Y_POSITION: -1, X: 5, TYPE: 5},
        ],
        columns=[STAFF_INDEX, Y_POSITION, X, TYPE])
    # Compare glyphs (rows in the glyphs array) regardless of their position in
    # the array (they are not required to be sorted).
    self.assertEqual(
        set(
            map(tuple,
                convolutional.Convolutional1DGlyphClassifier(
                    run_min_length=1)._build_detected_glyphs(
                        testing.PREDICTIONS))),
        set(map(tuple, glyphs.values)))

  def testNoGlyphs_dummyClassifier(self):

    class DummyClassifier(convolutional.Convolutional1DGlyphClassifier):
      """Outputs the classifications for no glyphs on multiple staves."""

      @property
      def staffline_predictions(self):
        return tf.fill([5, 9, 100], musicscore_pb2.Glyph.NONE)

    with self.test_session():
      self.assertAllEqual(
          DummyClassifier().get_detected_glyphs().eval(),
          np.zeros((0, 4), np.int32))


if __name__ == '__main__':
  tf.test.main()
