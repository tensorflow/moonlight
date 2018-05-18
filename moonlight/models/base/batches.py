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
"""Utility for batching and limiting the dataset size according to flags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('dataset_shuffle_buffer_size', 10000,
                     'Shuffles this many entries in the dataset. 0 indicates no'
                     ' shuffling.')
flags.DEFINE_integer(
    'dataset_limit_size', None,
    'Only take this many entries in the dataset (before repeating by'
    ' num_epochs).'
)
flags.DEFINE_integer('dataset_batch_size', 32, 'Resulting batch size.')
flags.DEFINE_integer('num_epochs', 1, 'Repeat the dataset by this number.')


def get_batched_tensor(dataset):
  """Gets the tensor representing a single batch from a `tf.data.Dataset`.

  Batch and epoch options are passed on the command line.

  Args:
    dataset: A `tf.data.Dataset` containing single examples.

  Returns:
    A dict of tensors, which contains the concatenated features from each
        example in a single batch. Each time the tensor is evaluated, it will
        produce the next batch.
  """
  if FLAGS.dataset_shuffle_buffer_size:
    dataset = dataset.shuffle(buffer_size=FLAGS.dataset_shuffle_buffer_size)
  if FLAGS.dataset_limit_size:
    dataset = dataset.take(FLAGS.dataset_limit_size)
  dataset = dataset.batch(FLAGS.dataset_batch_size)
  dataset = dataset.repeat(FLAGS.num_epochs)
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()
