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
"""Detects note beams.

Beams are long, very thick, horizontal or diagonal lines that may intersect with
the staves. To detect them, we use staff removal followed by extra binary
erosion, in case the staves are not completely removed and still have extra
black pixels around a beam. We then find all of the connected components,
because each beam should now be detached from the stem, staff, and (typically)
other beams. We filter beams by minimum width. Further processing and assignment
of stems to beams is done in `beam_processor.py`.

Each beam halves the duration of each note it is atteched to by a stem.
"""
# TODO(ringw): Make Hough line segments more robust, and then use them here
# instead of connected components.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from moonlight.structure import components
from moonlight.vision import morphology

COLUMNS = components.ConnectedComponentsColumns


class Beams(object):
  """Note beam detector."""

  def __init__(self, staff_remover, threshold=127):
    staff_detector = staff_remover.staff_detector
    image = morphology.binary_erosion(
        tf.less(staff_remover.remove_staves, threshold),
        staff_detector.staffline_thickness)
    beams = components.get_component_bounds(image)
    staffline_distance = tf.cond(
        tf.greater(tf.shape(staff_detector.staves)[0], 0),
        lambda: tf.reduce_mean(staff_detector.staffline_distance),
        lambda: tf.constant(0, tf.int32))
    min_length = 2 * staffline_distance
    keep_beam = tf.greater_equal(beams[:, COLUMNS.X1] - beams[:, COLUMNS.X0],
                                 min_length)
    keep_beam.set_shape([None])
    self.beams = tf.boolean_mask(beams, keep_beam)
    self.data = [self.beams]


class ComputedBeams(object):
  """Holder for the computed beams NumPy array."""

  def __init__(self, beams):
    self.beams = np.asarray(beams, np.int32)
