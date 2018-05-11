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
"""Base patch-based glyph model.

For example, this accepts the staff patch k-means centroids emitted by
staffline_patches_kmeans_pipeline and labeled by kmeans_labeler.

This defines the input and signature of the model, and allows any type of
multi-class classifier using the normalized patches as input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record

FLAGS = flags.FLAGS

flags.DEFINE_string('input_patches', None, 'Glob of labeled patch TFRecords')
flags.DEFINE_string('model_dir', None, 'Output trained model directory')
flags.DEFINE_integer(
    'num_epochs', 1, 'Number of passes to take over all patches')


def read_patch_dimensions():
  """Reads the dimensions of the input patches from disk.

  Parses the first example in the training set, which must have "height" and
  "width" features.

  Returns:
    Tuple of (height, width) read from disk, using the glob passed to
    --input_patches.
  """
  for filename in file_io.get_matching_files(FLAGS.input_patches):
    # If one matching file is empty, go on to the next file.
    for record in tf_record.tf_record_iterator(filename):
      example = tf.train.Example.FromString(record)
      # Convert long (int64) to int, necessary for use in feature columns in
      # Python 2.
      patch_height = int(example.features.feature['height'].int64_list.value[0])
      patch_width = int(example.features.feature['width'].int64_list.value[0])
      return patch_height, patch_width


def input_fn():
  """Defines the estimator input function.

  Returns:
    A callable. Each invocation returns a tuple containing:
    * A dict with a single key 'patch', and the patch tensor as a value.
    * A scalar tensor with the patch label, as an integer.
  """
  patch_height, patch_width = read_patch_dimensions()
  dataset = tf.data.TFRecordDataset(
      file_io.get_matching_files(FLAGS.input_patches))

  def parser(record):
    features = tf.parse_single_example(
        record, {
            'patch':
                tf.FixedLenFeature((patch_height, patch_width), tf.float32),
            'label':
                tf.FixedLenFeature((), tf.int64)
        })
    return {'patch': features['patch']}, features['label']

  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(FLAGS.num_epochs)
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


def serving_fn():
  """Returns the ServingInputReceiver for the exported model.

  Returns:
    A ServingInputReceiver object which may be passed to
    `Estimator.export_savedmodel`. A model saved using this receiver may be used
    for running OMR.
  """
  examples = tf.placeholder(tf.string, shape=[None])
  patch_height, patch_width = read_patch_dimensions()
  parsed = tf.parse_example(examples, {
      'patch': tf.FixedLenFeature((patch_height, patch_width), tf.float32),
  })
  return tf.estimator.export.ServingInputReceiver(
      features={'patch': parsed['patch']},
      receiver_tensors=parsed['patch'],
      receiver_tensors_alternatives={
          'example': examples,
          'patch': parsed['patch']
      })


def create_patch_feature_column():
  return tf.feature_column.numeric_column(
      'patch', shape=read_patch_dimensions())


# TODO(ringw): Evaluation.
def train_and_export(estimator):
  estimator.train(input_fn)
  estimator.export_savedmodel(FLAGS.model_dir, serving_fn)
