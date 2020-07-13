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
"""Applies noise to an image for generating training data.

All noise assumes a monochrome image with white (255) as background.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.contrib import image as contrib_image


def placeholder_image():
  return tf.placeholder(tf.uint8, shape=(None, None), name='placeholder_image')


def random_rotation(image, angle=math.pi / 180):
  return 255. - contrib_image.rotate(
      255. - tf.to_float(image),
      tf.random_uniform((), -angle, angle),
      interpolation='BILINEAR')


def gaussian_noise(image, stddev=5):
  return image + tf.random_normal(tf.shape(image), stddev=stddev)
