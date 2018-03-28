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
"""Tests for labeled data generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

from moonlight.staves import staffline_extractor
from moonlight.training.generation import generation

PATCH_WIDTH = 15


class GenerationTest(tf.test.TestCase):

  def testDoFn(self):
    page_gen = generation.PageGenerationDoFn(
        num_pages_per_batch=1,
        vexflow_generator_command=[
            os.path.join(tf.resource_loader.get_data_files_path(),
                         'vexflow_generator')
        ],
        svg_to_png_command=['/usr/bin/env', 'convert', 'svg:-', 'png:-'])
    page_gen.start_bundle()
    patch_examples = generation.PatchExampleDoFn(
        negative_example_distance=5,
        patch_width=PATCH_WIDTH,
        negative_to_positive_example_ratio=1.0)
    patch_examples.start_bundle()
    examples = [
        example
        for page in page_gen.process(0)
        for example in patch_examples.process(page)
    ]
    page_gen.finish_bundle()
    patch_examples.finish_bundle()
    self.assertGreater(len(examples), 4)
    for example in examples:
      self.assertEqual(
          len(example.features.feature['patch'].float_list.value),
          PATCH_WIDTH * staffline_extractor.DEFAULT_TARGET_HEIGHT)


if __name__ == '__main__':
  tf.test.main()
