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
        flags.FLAGS.augmentation_x_shift_probability = 0
        flags.FLAGS.augmentation_max_rotation_degrees = 0
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

      flags.FLAGS.train_input_patches = records_file.name
      batch_tensors = glyph_patches.input_fn(records_file.name)

      with self.test_session() as sess:
        batch = sess.run(batch_tensors)

        self.assertAllEqual(
            batch[0]['patch'],
            np.arange(height * width).reshape((1, height, width)).repeat(
                3, axis=0))
        self.assertAllEqual(batch[1], [label, label, label])

  def testShiftLeft(self):
    with self.test_session():
      self.assertAllEqual(
          # pyformat: disable
          glyph_patches._shift_left([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10,
                                                                  11]]).eval(),
          [[1, 2, 3, 3], [5, 6, 7, 7], [9, 10, 11, 11]])

  def testShiftRight(self):
    with self.test_session():
      self.assertAllEqual(
          # pyformat: disable
          glyph_patches._shift_right([[0, 1, 2, 3], [4, 5, 6, 7],
                                      [8, 9, 10, 11]]).eval(),
          [[0, 0, 1, 2], [4, 4, 5, 6], [8, 8, 9, 10]])

  def testMulticlassBinaryMetric(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    labels =                     tf.constant([1, 1, 3, 2, 2, 2, 2])
    predictions = dict(class_ids=tf.constant([1, 3, 2, 2, 4, 3, 2]))
    _, precision_1 = glyph_patches.multiclass_binary_metric(
        1, tf.metrics.precision, labels, predictions)
    _, recall_1 = glyph_patches.multiclass_binary_metric(
        1, tf.metrics.recall, labels, predictions)
    _, precision_2 = glyph_patches.multiclass_binary_metric(
        2, tf.metrics.precision, labels, predictions)
    _, recall_2 = glyph_patches.multiclass_binary_metric(
        2, tf.metrics.recall, labels, predictions)
    _, precision_3 = glyph_patches.multiclass_binary_metric(
        3, tf.metrics.precision, labels, predictions)
    _, recall_3 = glyph_patches.multiclass_binary_metric(
        3, tf.metrics.recall, labels, predictions)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      # For class 1: 1 true positive and no false positives
      self.assertEqual(1.0, precision_1.eval())
      # For class 1: 1 true positive and 1 false negative
      self.assertEqual(0.5, recall_1.eval())
      # For class 2: 2 true positives and 1 false positive
      self.assertAlmostEqual(2 / 3, precision_2.eval(), places=5)
      # For class 2: 2 true positives and 2 false negatives
      self.assertEqual(0.5, recall_2.eval())
      # For class 3: No true positives
      self.assertEqual(0, precision_3.eval())
      self.assertEqual(0, recall_3.eval())


if __name__ == '__main__':
  tf.test.main()
