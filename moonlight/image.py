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
"""Utility for reading music score images.

Reads grayscale images, and reverses the values if the image is detected to be
inverted.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def decode_music_score_png(contents):
  """Reads a music score image.

  This reads a binary or grayscale image and takes the only channel. If the
      image is detected to be inverted, the values will be flipped so that
      the white background has value 255 and the black content has value 0.

  Args:
    contents: PNG data in a scalar string tensor.

  Returns:
    The music score image. A two-dimensional tensor (HW) of type uint8.
  """
  with tf.name_scope("decode_music_score_png"):
    contents = tf.convert_to_tensor(contents, name="contents")
    image_t = tf.image.decode_png(contents, channels=1, dtype=tf.uint8)[:, :, 0]

    def inverted_image():
      # Sub op is not defined for uint8.
      int32_image = tf.cast(image_t, tf.int32)
      return tf.cast(255 - int32_image, tf.uint8)

    threshold = 127
    num_pixels = tf.shape(image_t)[0] * tf.shape(image_t)[1]
    majority_dark = tf.greater(
        tf.reduce_sum(tf.cast(image_t < threshold, tf.int32)), num_pixels // 2)
    return tf.cond(majority_dark, inverted_image, lambda: image_t)
