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
"""Tests for glyph_patches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl import flags
from moonlight.models.base import glyph_patches
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import tf_record


class GlyphPatchesTest(tf.test.TestCase):

  def testInputFn(self):
    with tempfile.NamedTemporaryFile() as records_file:
      with tf_record.TFRecordWriter(records_file.name) as records_writer:
        example = tf.train.Example()
        height = 5
        width = 3
        example.features.feature['height'].int64_list.value.append(height)
        example.features.feature['width'].int64_list.value.append(width)
        example.features.feature['patch'].float_list.value.extend(
            range(height * width))
        label = 1
        example.features.feature['label'].int64_list.value.append(label)
        for _ in range(3):
          records_writer.write(example.SerializeToString())

      flags.FLAGS.input_patches = records_file.name
      batch_tensors = glyph_patches.input_fn()

      with self.test_session() as sess:
        batch = sess.run(batch_tensors)

        self.assertAllEqual(
            batch[0]['patch'],
            np.arange(height * width).reshape((1, height, width)).repeat(
                3, axis=0))
        self.assertAllEqual(batch[1], [label, label, label])


if __name__ == '__main__':
  tf.test.main()
