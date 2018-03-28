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
"""Tests for the KNN glyph classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import pandas as pd
import tensorflow as tf

from moonlight.glyphs import base
from moonlight.glyphs import knn
from moonlight.protobuf import musicscore_pb2

STAFF_INDEX = base.GlyphsTensorColumns.STAFF_INDEX
Y_POSITION = base.GlyphsTensorColumns.Y_POSITION
X = base.GlyphsTensorColumns.X
TYPE = base.GlyphsTensorColumns.TYPE
Glyph = musicscore_pb2.Glyph  # pylint: disable=invalid-name


class KnnTest(tf.test.TestCase):

  def testFakeStaffline(self):
    # Staffline containing fake glyphs.
    staffline = tf.constant(
        [[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]])
    staffline = tf.cast(staffline, tf.float32)
    # Mapping of glyphs (row-major order) to glyph types.
    patterns = {
        (1, 0, 1, 0, 1, 0, 1, 0, 1): musicscore_pb2.Glyph.CLEF_TREBLE,
        (0, 0, 0, 0, 1, 0, 0, 0, 0): musicscore_pb2.Glyph.NOTEHEAD_FILLED,
        (1, 1, 1, 1, 0, 1, 1, 1, 1): musicscore_pb2.Glyph.NOTEHEAD_EMPTY,
        (0, 0, 0, 0, 0, 0, 0, 0, 0): musicscore_pb2.Glyph.NONE,
        (1, 0, 0, 0, 0, 0, 0, 0, 0): musicscore_pb2.Glyph.NONE,
        (0, 1, 0, 0, 0, 0, 0, 0, 0): musicscore_pb2.Glyph.NONE,
        (0, 0, 1, 0, 0, 0, 0, 0, 0): musicscore_pb2.Glyph.NONE,
        (0, 0, 0, 1, 0, 0, 0, 0, 0): musicscore_pb2.Glyph.NONE,
        (0, 0, 0, 0, 0, 1, 0, 0, 0): musicscore_pb2.Glyph.NONE,
        (0, 0, 0, 0, 0, 0, 1, 0, 0): musicscore_pb2.Glyph.NONE,
        (0, 0, 0, 0, 0, 0, 0, 1, 0): musicscore_pb2.Glyph.NONE,
        (0, 0, 0, 0, 0, 0, 0, 0, 1): musicscore_pb2.Glyph.NONE,
    }
    with tf.Session():
      with tempfile.NamedTemporaryFile(mode='r') as examples_file:
        with tf.python_io.TFRecordWriter(examples_file.name) as writer:
          # Sort the keys for determinism.
          for pattern in sorted(patterns):
            example = tf.train.Example()
            example.features.feature['patch'].float_list.value.extend(pattern)
            example.features.feature['height'].int64_list.value.append(3)
            example.features.feature['width'].int64_list.value.append(3)
            example.features.feature['label'].int64_list.value.append(
                patterns[pattern])
            writer.write(example.SerializeToString())

        class FakeStafflineExtractor(object):

          def extract_staves(self):
            return staffline[None, None, :, :]

        # stafflines are 4D (num_staves, num_stafflines, height, width).
        classifier = knn.NearestNeighborGlyphClassifier(
            examples_file.name, FakeStafflineExtractor(), run_min_length=1)
        k_nearest_value = tf.get_default_graph().get_tensor_by_name(
            'k_nearest_value:0')
        glyphs = classifier.get_detected_glyphs().eval(feed_dict={
            k_nearest_value: 1
        })
    # The patches of the staffline that match non-NONE patterns (in row-major
    # order) should appear here (x is their center coordinate).
    # pyformat: disable
    expected_glyphs = pd.DataFrame(
        [
            {STAFF_INDEX: 0, Y_POSITION: 0, X: 1, TYPE: Glyph.CLEF_TREBLE},
            {STAFF_INDEX: 0, Y_POSITION: 0, X: 9, TYPE: Glyph.NOTEHEAD_EMPTY},
            {STAFF_INDEX: 0, Y_POSITION: 0, X: 14, TYPE: Glyph.CLEF_TREBLE},
            {STAFF_INDEX: 0, Y_POSITION: 0, X: 19, TYPE: Glyph.NOTEHEAD_FILLED},
        ],
        columns=[STAFF_INDEX, Y_POSITION, X, TYPE])
    self.assertAllEqual(glyphs, expected_glyphs)


if __name__ == '__main__':
  tf.test.main()
