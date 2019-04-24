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
"""A pipeline for generating labeled patch data from VexFlow.

See `generation.py` for details on the output data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import apache_beam as beam
from moonlight.pipeline import pipeline_flags
from moonlight.training.generation import generation
from moonlight.training.generation import image_noise
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_positive_examples', 1000000,
                     'The number of positive examples to generate.')
flags.DEFINE_string('examples_path', '', 'The path of the output examples.')
flags.DEFINE_integer('num_shards', None, 'Fixed number of shards (optional)')
flags.DEFINE_multi_string('vexflow_generator_command', [
    '/usr/bin/env',
    'node',
    'vexflow_generator.js',
], 'Command line to run the node.js vexflow generator.')
flags.DEFINE_float('negative_to_positive_example_ratio', 10,
                   'Ratio of negative to positive examples.')
flags.DEFINE_integer('num_pages_per_batch', 100,
                     'The number of pages to emit in every node.js run.')
flags.DEFINE_multi_string(
    'svg_to_png_command', [
        '/usr/bin/env',
        'convert',
        'svg:-',
        'png:-',
    ], 'Command line to convert a SVG (stdin) to PNG (stdout).')
flags.DEFINE_integer('patch_width', 15, 'Width of a staffline patch.')
flags.DEFINE_integer(
    'negative_example_distance', 3,
    'The minimum distance of a negative example from any glyph.')


def main(_):
  with pipeline_flags.create_pipeline() as pipeline:
    num_pages = (FLAGS.num_positive_examples +
                 generation.POSITIVE_EXAMPLES_PER_IMAGE -
                 1) // generation.POSITIVE_EXAMPLES_PER_IMAGE
    num_batches = (num_pages + FLAGS.num_pages_per_batch -
                   1) // FLAGS.num_pages_per_batch
    batch_nums = pipeline | beam.transforms.Create(list(range(num_batches)))
    pages = batch_nums | beam.ParDo(
        generation.PageGenerationDoFn(
            num_pages_per_batch=FLAGS.num_pages_per_batch,
            vexflow_generator_command=FLAGS.vexflow_generator_command,
            svg_to_png_command=FLAGS.svg_to_png_command))

    def noise_fn(image):
      # TODO(ringw): Add better noise, maybe using generative adversarial
      # networks trained on real scores from IMSLP.
      return image_noise.gaussian_noise(image_noise.random_rotation(image))

    examples = pages | beam.ParDo(
        generation.PatchExampleDoFn(
            negative_example_distance=FLAGS.negative_example_distance,
            patch_width=FLAGS.patch_width,
            negative_to_positive_example_ratio=FLAGS
            .negative_to_positive_example_ratio,
            noise_fn=noise_fn))
    examples |= beam.io.WriteToTFRecord(
        FLAGS.examples_path,
        beam.coders.ProtoCoder(tf.train.Example),
        num_shards=FLAGS.num_shards)


if __name__ == '__main__':
  app.run(main)
