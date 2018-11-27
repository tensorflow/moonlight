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

import itertools
import logging

import apache_beam as beam
from apache_beam import metrics
from moonlight.staves import staffline_extractor
from moonlight.util import more_iter_tools
import numpy as np
import tensorflow as tf


def _filter_patch(patch, min_num_dark_pixels=10):
  unused_patch_name, patch = patch
  return np.greater_equal(np.sum(np.less(patch, 0.5)), min_num_dark_pixels)


class StafflinePatchesDoFn(beam.DoFn):
  """Runs the staffline patches graph."""

  def __init__(self, patch_height, patch_width, num_stafflines, timeout_ms,
               max_patches_per_page):
    self.patch_height = patch_height
    self.patch_width = patch_width
    self.num_stafflines = num_stafflines
    self.timeout_ms = timeout_ms
    self.max_patches_per_page = max_patches_per_page
    self.total_pages_counter = metrics.Metrics.counter(self.__class__,
                                                       'total_pages')
    self.failed_pages_counter = metrics.Metrics.counter(self.__class__,
                                                        'failed_pages')
    self.successful_pages_counter = metrics.Metrics.counter(
        self.__class__, 'successful_pages')
    self.empty_pages_counter = metrics.Metrics.counter(self.__class__,
                                                       'empty_pages')
    self.total_patches_counter = metrics.Metrics.counter(
        self.__class__, 'total_patches')
    self.emitted_patches_counter = metrics.Metrics.counter(
        self.__class__, 'emitted_patches')

  def start_bundle(self):
    self.extractor = staffline_extractor.StafflinePatchExtractor(
        patch_height=self.patch_height,
        patch_width=self.patch_width,
        run_options=tf.RunOptions(timeout_in_ms=self.timeout_ms))
    self.session = tf.Session(graph=self.extractor.graph)

  def process(self, png_path):
    self.total_pages_counter.inc()
    try:
      with self.session.as_default():
        patches_iter = self.extractor.page_patch_iterator(png_path)
    # pylint: disable=broad-except
    except Exception:
      logging.exception('Skipping failed music score (%s)', png_path)
      self.failed_pages_counter.inc()
      return
    patches_iter = itertools.ifilter(_filter_patch, patches_iter)

    if 0 < self.max_patches_per_page:
      # Subsample patches.
      patches = more_iter_tools.iter_sample(patches_iter,
                                            self.max_patches_per_page)
    else:
      patches = list(patches_iter)

    if not patches:
      self.empty_pages_counter.inc()
    self.total_patches_counter.inc(len(patches))

    # Serialize each patch as an Example.
    for patch_name, patch in patches:
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
    del self.extractor
    del self.session
