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
"""Extracts non-empty patches of extracted stafflines.

Extracts vertical slices of the image where glyphs are expected
(see `staffline_extractor.py`), and takes horizontal windows of the slice which
will be clustered. Some patches will have a glyph roughly in their center, and
the corresponding cluster centroids will be labeled as such.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path
import random

import apache_beam as beam
from apache_beam import metrics
import tensorflow as tf

from moonlight import image
from moonlight import staves
from moonlight.staves import removal
from moonlight.staves import staffline_extractor
from moonlight.util import patches as util_patches


def pipeline_graph(png_path, staffline_height, patch_width, num_stafflines):
  """Constructs the graph for the staffline patches pipeline.

  Args:
    png_path: Path to the input png. String scalar tensor.
    staffline_height: Height of a staffline. int.
    patch_width: Width of a patch. int.
    num_stafflines: Number of stafflines to extract around each staff. int.

  Returns:
    A tensor representing the staffline patches. float32 with shape
        (num_patches, staffline_height, patch_width).
  """
  image_t = image.decode_music_score_png(tf.read_file(png_path))
  staff_detector = staves.StaffDetector(image_t)
  staff_remover = removal.StaffRemover(staff_detector)
  stafflines = tf.identity(
      staffline_extractor.StafflineExtractor(
          staff_remover.remove_staves,
          staff_detector,
          target_height=staffline_height,
          num_sections=num_stafflines).extract_staves(),
      name='stafflines')
  return _extract_patches(stafflines, patch_width)


def _extract_patches(stafflines, patch_width, min_num_dark_pixels=10):
  patches = util_patches.patches_1d(stafflines, patch_width)
  # Limit to patches that have min_num_dark_pixels.
  num_dark_pixels = tf.reduce_sum(
      tf.where(
          tf.less(patches, 0.5),
          tf.ones_like(patches, dtype=tf.int32),
          tf.zeros_like(patches, dtype=tf.int32)),
      axis=[-2, -1])
  return tf.boolean_mask(patches,
                         tf.greater_equal(num_dark_pixels, min_num_dark_pixels))


class StafflinePatchesDoFn(beam.DoFn):
  """Runs the staffline patches graph."""

  def __init__(self, patch_height, patch_width, num_stafflines, timeout_ms,
               max_patches_per_page):
    self.patch_height = patch_height
    self.patch_width = patch_width
    self.num_stafflines = num_stafflines
    self.timeout_ms = timeout_ms
    self.max_patches_per_page = max_patches_per_page
    self.total_pages_counter = metrics.Metrics.counter(
        self.__class__, 'total_pages')
    self.failed_pages_counter = metrics.Metrics.counter(
        self.__class__, 'failed_pages')
    self.successful_pages_counter = metrics.Metrics.counter(
        self.__class__, 'successful_pages')
    self.empty_pages_counter = metrics.Metrics.counter(
        self.__class__, 'empty_pages')
    self.total_patches_counter = metrics.Metrics.counter(
        self.__class__, 'total_patches')
    self.emitted_patches_counter = metrics.Metrics.counter(
        self.__class__, 'emitted_patches')

  def start_bundle(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.session = tf.Session()
      with self.session.as_default():
        # Construct the graph.
        self.png_path = tf.placeholder(tf.string, shape=(), name='png_path')
        self.patches = pipeline_graph(self.png_path, self.patch_height,
                                      self.patch_width, self.num_stafflines)

  def process(self, png_path):
    self.total_pages_counter.inc()
    basename = os.path.basename(png_path)
    run_options = tf.RunOptions(timeout_in_ms=self.timeout_ms)
    try:
      patches = self.session.run(
          self.patches,
          feed_dict={self.png_path: png_path},
          options=run_options)
    # pylint: disable=broad-except
    except Exception:
      logging.exception('Skipping failed music score (%s)', png_path)
      self.failed_pages_counter.inc()
      return

    # len() is required for NumPy ndarrays.
    # pylint: disable=g-explicit-length-test
    if not len(patches):
      self.empty_pages_counter.inc()
    self.total_patches_counter.inc(len(patches))

    # Subsample patches.
    if 0 < self.max_patches_per_page < len(patches):
      patch_inds = random.sample(
          xrange(len(patches)), self.max_patches_per_page)
      patches = patches[patch_inds]
    else:
      # Patches numbered 0 through n - 1.
      patch_inds = range(len(patches))
    # Serialize each patch as an Example. The index uniquely identifies the
    # patch.
    for ind, patch in zip(patch_inds, patches):
      patch_name = (basename + '#' + str(ind)).encode('utf-8')
      example = tf.train.Example()
      example.features.feature['name'].bytes_list.value.append(patch_name)
      example.features.feature['features'].float_list.value.extend(
          patch.ravel())
      example.features.feature['height'].int64_list.value.append(patch.shape[0])
      example.features.feature['width'].int64_list.value.append(patch.shape[1])
      yield example

    self.successful_pages_counter.inc()
    # Patches are sub-sampled by this point.
    self.emitted_patches_counter.inc(len(patches))

  def finish_bundle(self):
    self.session.close()
    del self.session
