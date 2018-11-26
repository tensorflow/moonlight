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
"""Tests for batches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from moonlight.models.base import batches
import numpy as np
import tensorflow as tf


class BatchesTest(tf.test.TestCase):

  def testBatching(self):
    all_as = np.random.rand(1000, 2, 3)
    all_bs = np.random.randint(0, 100, [1000], np.int32)
    all_labels = np.random.randint(0, 5, [1000], np.int32)
    random_dataset = tf.data.Dataset.from_tensor_slices(({
        'a': tf.constant(all_as),
        'b': tf.constant(all_bs)
    }, tf.constant(all_labels)))

    flags.FLAGS.dataset_shuffle_buffer_size = 0
    batch_tensors = batches.get_batched_tensor(random_dataset)
    with self.test_session() as sess:
      batch = sess.run(batch_tensors)

      # First batch.
      self.assertEqual(len(batch), 2)
      self.assertEqual(sorted(batch[0].keys()), ['a', 'b'])
      batch_size = flags.FLAGS.dataset_batch_size
      self.assertAllEqual(batch[0]['a'], all_as[:batch_size])
      self.assertAllEqual(batch[0]['b'], all_bs[:batch_size])
      self.assertAllEqual(batch[1], all_labels[:batch_size])

      batch = sess.run(batch_tensors)

      # Second batch.
      self.assertEqual(len(batch), 2)
      self.assertEqual(sorted(batch[0].keys()), ['a', 'b'])
      batch_size = flags.FLAGS.dataset_batch_size
      self.assertAllEqual(batch[0]['a'], all_as[batch_size:batch_size * 2])
      self.assertAllEqual(batch[0]['b'], all_bs[batch_size:batch_size * 2])
      self.assertAllEqual(batch[1], all_labels[batch_size:batch_size * 2])


if __name__ == '__main__':
  tf.test.main()
