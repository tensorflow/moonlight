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
"""Tests for the staffline patches DoFn graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tempfile

from absl.testing import absltest
import apache_beam as beam
import tensorflow as tf
from tensorflow.python.lib.io import tf_record

from moonlight.staves import staffline_extractor
from moonlight.training.clustering import staffline_patches_dofn

PATCH_HEIGHT = 9
PATCH_WIDTH = 7
NUM_STAFFLINES = 9
TIMEOUT_MS = 60000
MAX_PATCHES_PER_PAGE = 10


class StafflinePatchesDoFnTest(absltest.TestCase):

  def testPipeline_corpusImage(self):
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../../testdata/IMSLP00747-000.png')
    with tempfile.NamedTemporaryFile() as output_examples:
      # Run the pipeline to get the staffline patches.
      with beam.Pipeline() as pipeline:
        dofn = staffline_patches_dofn.StafflinePatchesDoFn(
            PATCH_HEIGHT, PATCH_WIDTH, NUM_STAFFLINES, TIMEOUT_MS,
            MAX_PATCHES_PER_PAGE)
        # pylint: disable=expression-not-assigned
        (pipeline | beam.transforms.Create([filename])
         | beam.transforms.ParDo(dofn) | beam.io.WriteToTFRecord(
             output_examples.name,
             beam.coders.ProtoCoder(tf.train.Example),
             shard_name_template=''))
      # Get the staffline images from a local TensorFlow session.
      extractor = staffline_extractor.StafflinePatchExtractor()
      with tf.Session(graph=extractor.graph):
        expected_patches = [
            tuple(patch.ravel())
            for unused_key, patch in extractor.page_patch_iterator(filename)
        ]
      for example_bytes in tf_record.tf_record_iterator(output_examples.name):
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        patch_pixels = tuple(
            example.features.feature['features'].float_list.value)
        self.assertIn(patch_pixels, expected_patches)


if __name__ == '__main__':
  absltest.main()
