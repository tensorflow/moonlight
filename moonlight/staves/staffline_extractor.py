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
"""Extracts horizontal slices from a staff for glyph classification."""
# TODO(ringw): Rename StafflineExtractor to PositionExtractor. Stafflines in
# this context should be renamed "extracted positions" to avoid confusion.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
from moonlight import image as image_module
from moonlight import staves as staves_module
from moonlight.staves import removal
import tensorflow as tf

DEFAULT_TARGET_HEIGHT = 18
DEFAULT_NUM_SECTIONS = 19
DEFAULT_STAFFLINE_DISTANCE_MULTIPLE = 3


class Axes(enum.IntEnum):

  STAFF = 0
  POSITION = 1
  Y = 2
  X = 3


def get_staffline(y_position, extracted_staff_arr):
  """Gets the staffline of the extracted staff.

  Args:
    y_position: The staffline position--the relative number of notes from the
        3rd line on the staff.
    extracted_staff_arr: An extracted staff NumPy array, e.g.
        `StafflineExtractor.extract_staves()[0].eval()` (`StafflineExtractor`
        returns multiple staves).

  Returns:
    The correct staffline from `extracted_staff_arr`, with shape
        `(target_height, image_width)`.

  Raises:
    ValueError: If the `y_position` is out of bounds in either direction.
  """
  return extracted_staff_arr[y_position_to_index(y_position,
                                                 len(extracted_staff_arr))]


def y_position_to_index(y_position, num_stafflines):
  index = num_stafflines // 2 - y_position
  if not 0 <= index < num_stafflines:
    raise ValueError('y_position %d too large for %d stafflines' %
                     (y_position, num_stafflines))
  return index


class StafflineExtractor(object):
  """Extracts horizontal slices from a staff for glyph classification.

  Glyphs must be centered on either a staff line or a staff space (halfway
  between staff lines). For classification, a window is extracted with height
  2*staffline_distance around a staffline or staff space. If num_sections is 9,
  extracts the five staff lines and the staff spaces between them.

  The slice is scaled proportionally to the staffline distance, making the
  output height equal to target_height, so that the glyph classifier is
  scale-invariant.
  """

  def __init__(self, image, staves,
               target_height=DEFAULT_TARGET_HEIGHT,
               num_sections=DEFAULT_NUM_SECTIONS,
               staffline_distance_multiple=DEFAULT_STAFFLINE_DISTANCE_MULTIPLE):
    """Create the staffline extractor.

    Args:
      image: A uint8 tensor of shape (height, width). The background (usually
          white) must have a value of 0.
      staves: An instance of base.BaseStaffDetector.
      target_height: The height of the scaled output windows.
      num_sections: The number of stafflines to extract.
      staffline_distance_multiple: The height of the extracted staffline, in
          multiples of the staffline distance. For example, a notehead should
          fit in a staffline distance multiple of 1, because it starts and ends
          vertically on a staff line. However, other glyphs may need more space
          above and below to classify accurately.
    """
    self.float_image = tf.cast(image, tf.float32) / 255.
    self.staves = staves
    self.target_height = target_height
    self.num_sections = num_sections
    self.staffline_distance_multiple = staffline_distance_multiple

    # Calculate the maximum width needed.
    min_staffline_distance = tf.reduce_min(staves.staffline_distance)
    self.target_width = self._get_resized_width(min_staffline_distance)

  def extract_staves(self):
    """Extracts stafflines from all staves in the image.

    Returns:
      A float32 Tensor of shape
      (num_staves, num_sections, target_height, slice_width). If the staffline
      distance is inconsistent between staves, smaller staves will be padded
      on the right with zeros.
    """
    # Only map if we have any staves, otherwise return an empty array with the
    # correct dimensionality.
    def do_extract_staves():
      """Actually performs staffline extraction if we have any staves.

      Returns:
        The stafflines tensor. See outer function doc.
      """
      staff_ys = self.staves.staves_interpolated_y

      def extract_staff(i):
        def extract_staffline_by_index(j):
          return self._extract_staffline(
              staff_ys[i], self.staves.staffline_distance[i], j)
        return tf.map_fn(
            extract_staffline_by_index,
            tf.range(-(self.num_sections // 2), self.num_sections // 2 + 1),
            dtype=tf.float32)

      return tf.map_fn(
          extract_staff,
          tf.range(tf.shape(self.staves.staves)[0]),
          dtype=tf.float32)

    # Shape of the empty stafflines tensor, if no staves are present.
    empty_shape = (0, self.num_sections, self.target_height, 0)
    stafflines = tf.cond(
        tf.shape(self.staves.staves)[0] > 0,
        do_extract_staves,
        # Otherwise, return an empty stafflines array.
        lambda: tf.zeros(empty_shape, tf.float32))
    # We need target_height to be statically known for e.g. `util/patches.py`.
    stafflines.set_shape((None, self.num_sections, self.target_height, None))
    return stafflines

  def _extract_staffline(self, staff_y, staffline_distance, staffline_num):
    """Extracts a single staffline from a single staff."""
    # Use a float image on a 0.0-1.0 scale for classification.
    image_shape = tf.shape(self.float_image)
    height = image_shape[0]  # Can't unpack a tensor object.
    width = image_shape[1]

    # Calculate the height of the extracted staffline in the unscaled image.
    staff_window = self._get_staffline_window_size(staffline_distance)

    # Calculate the coordinates to extract for the window.
    # Note: tf.meshgrid uses xs before ys by default, but y is the 0th axis
    # for indexing.
    xs, ys = tf.meshgrid(
        tf.range(width), tf.range(staff_window) - (staff_window // 2))
    # ys are centered around 0. Add the staff_y, repeating along the
    # 0th axis.
    ys += tf.tile(staff_y[None, :], [staff_window, 1])
    # Add the offset for the staff line within the staff.
    # Round up in case the y position is not whole (in between staff lines with
    # an odd staffline distance). This puts the center of the staff space closer
    # to the center of the window.
    ys += tf.cast(
        tf.ceil(tf.truediv(staffline_num * staffline_distance, 2)), tf.int32)

    invalid = tf.logical_not(
        (0 <= ys) & (ys < height) & (0 <= xs) & (xs < width))
    # Use a coordinate of (0, 0) for pixels outside of the original image.
    # We will then fill in those pixels with zeros.
    ys = tf.where(invalid, tf.zeros_like(ys), ys)
    xs = tf.where(invalid, tf.zeros_like(xs), xs)
    inds = tf.stack([ys, xs], axis=2)
    staffline_image = tf.gather_nd(self.float_image, inds)
    # Fill the pixels outside of the original image with zeros.
    staffline_image = tf.where(
        invalid, tf.zeros_like(staffline_image), staffline_image)

    # Calculate the proportional width after scaling the height to
    # self.target_height.
    resized_width = self._get_resized_width(staffline_distance)
    # Use area resizing because we expect the output to be smaller.
    # Add extra axes, because we only have 1 image and 1 channel.
    staffline_image = tf.image.resize_area(
        staffline_image[None, :, :, None],
        [self.target_height, resized_width])[0, :, :, 0]
    # Pad to make the width consistent with target_width.
    staffline_image = tf.pad(staffline_image,
                             [[0, 0], [0, self.target_width - resized_width]])
    return staffline_image

  def _get_resized_width(self, staffline_distance):
    image_width = tf.shape(self.float_image)[1]
    window_height = self._get_staffline_window_size(staffline_distance)
    return tf.cast(
        tf.round(tf.truediv(image_width * self.target_height, window_height)),
        tf.int32)

  def _get_staffline_window_size(self, staffline_distance):
    return tf.to_int32(
        tf.round(
            tf.to_float(staffline_distance) * tf.to_float(
                self.staffline_distance_multiple)))


class StafflinePatchExtractor(object):
  """Wraps the OMR TensorFlow graph and performs staff patch extraction.

  Extracts a single patch from an image, to be used for training.
  """

  def __init__(self,
               num_sections=DEFAULT_NUM_SECTIONS,
               patch_height=15,
               patch_width=12):
    self.num_sections = num_sections
    self.patch_height = patch_height
    self.patch_width = patch_width

    self.graph = tf.Graph()
    with self.graph.as_default():
      # Identifying information for the patch.
      self.filename = tf.placeholder(tf.string)
      self.staff_index = tf.placeholder(tf.int64)
      self.y_position = tf.placeholder(tf.int64)

      image = image_module.decode_music_score_png(tf.read_file(self.filename))
      staff_detector = staves_module.StaffDetector(image)
      staff_remover = removal.StaffRemover(staff_detector)
      extractor = StafflineExtractor(
          staff_remover.remove_staves,
          staff_detector,
          num_sections=num_sections,
          target_height=patch_height)
      # Index into the staff strips array, where a y position of 0 is the center
      # element. Positive positions count up (towards higher notes, towards the
      # top of the image, and smaller indices into the array).
      position_index = num_sections // 2 - self.y_position
      # The entire extracted horizontal strip of the image.
      self.staffline = extractor.extract_staves()[self.staff_index,
                                                  position_index]

      # Determine the scale for converting image x coordinates to the scaled
      # staff strip from which the patch is extracted.
      extracted_staff_strip_height = tf.shape(self.staffline)[0]
      staffline_distance = staff_detector.staffline_distance[self.staff_index]
      unscaled_staff_strip_height = tf.multiply(
          DEFAULT_STAFFLINE_DISTANCE_MULTIPLE, staffline_distance)
      self.staffline_scale = tf.divide(
          tf.to_float(extracted_staff_strip_height),
          tf.to_float(unscaled_staff_strip_height))

  def extract_staff_strip(self, filename, staff_index, y_position):
    """Extracts an entire horizontal strip from the image.

    Args:
      filename: The absolute filename of the image.
      staff_index: Index of the staff out of all staves on the page.
      y_position: Note y position on the staff, on which to extract the strip.
          The position starts out 0 for the staff center line, and grows more
          positive for higher notes.

    Returns:
      A tuple of:
        A wide strip of the image as a NumPy array.
        The scale factor from the original image scale to the normalized staff
            strip scale.
    """
    return tf.get_default_session().run(
        [self.staffline, self.staffline_scale],
        feed_dict={
            self.filename: filename,
            self.staff_index: staff_index,
            self.y_position: y_position,
        })

  def extract_staff_patch(self, filename, staff_index, y_position, image_x):
    """Extracts a rectangular patch to be labeled.

    Args:
      filename: The absolute filename of the image.
      staff_index: Index of the staff out of all staves on the page.
      y_position: Note y position on the staff, on which to extract the strip.
      image_x: The coordinate of the patch center x, in image coordinates.

    Returns:
      The rectangular NumPy array for the patch.

    Raises:
      ValueError: If the x coordinate is too close to the left or right edge of
          the image to extract a full patch.
    """
    staffline, scale = self.extract_staff_strip(filename, staff_index,
                                                y_position)
    staffline_x = int(round(image_x * scale))
    patch_x_start = staffline_x + (-self.patch_width // 2)
    patch_x_stop = staffline_x + self.patch_width // 2
    if not (self.patch_width // 2 <= staffline_x <
            staffline.shape[1] - self.patch_width // 2):
      raise ValueError('image_x too close to bounds of image')
    return staffline[:, patch_x_start:patch_x_stop]
