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
"""Tool to convert existing KNN tfrecords to a saved model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import tensorflow as tf

from moonlight.glyphs import corpus
from moonlight.glyphs import knn_model


def run(tfrecords_filename, export_dir):
  with tf.Session():
    height, width = corpus.get_patch_shape(tfrecords_filename)
    patches, labels = corpus.parse_corpus(tfrecords_filename, height, width)

  knn_model.export_knn_model(patches, labels, export_dir)


def main(argv):
  _, infile, outdir = argv
  run(infile, outdir)


if __name__ == '__main__':
  app.run(main)
