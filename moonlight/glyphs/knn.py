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
"""K-Nearest-Neighbors glyph classification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from moonlight.glyphs import convolutional
from moonlight.glyphs import corpus
from moonlight.protobuf import musicscore_pb2
from moonlight.util import patches

# k = 3 has the best performance for noteheads, clefs, and sharps. k = 5 seems
# to increase false negatives, so we probably don't want to increase k further
# with our current data.
K_NEAREST_VALUE = 3

NUM_GLYPHS = len(musicscore_pb2.Glyph.Type.keys())


class NearestNeighborGlyphClassifier(
    convolutional.Convolutional1DGlyphClassifier):
  """Classifies staffline patches using 1 nearest neighbor."""

  def __init__(self, corpus_file, staffline_extractor, **kwargs):
    """Build a 1-nearest-neighbor classifier with labeled patches.

    Args:
      corpus_file: Path to the TFRecords of Examples with patch (cluster) values
        in the "patch" feature, and the glyph label in the "label" feature.
      staffline_extractor: The staffline extractor.
      **kwargs: Passed through to `Convolutional1DGlyphClassifier`.
    """
    super(NearestNeighborGlyphClassifier, self).__init__(**kwargs)

    patch_height, patch_width = corpus.get_patch_shape(corpus_file)
    centroids, labels = corpus.parse_corpus(corpus_file, patch_height,
                                            patch_width)
    centroids_shape = tf.shape(centroids)
    flattened_centroids = tf.reshape(
        centroids,
        [centroids_shape[0], centroids_shape[1] * centroids_shape[2]])
    self.staffline_extractor = staffline_extractor
    stafflines = staffline_extractor.extract_staves()
    # Collapse the stafflines per stave.
    width = tf.shape(stafflines)[-1]
    # Shape (num_staves, num_stafflines, num_patches, height, patch_width).
    staffline_patches = patches.patches_1d(stafflines, patch_width)
    staffline_patches_shape = tf.shape(staffline_patches)
    flattened_patches = tf.reshape(staffline_patches, [
        staffline_patches_shape[0] * staffline_patches_shape[1] *
        staffline_patches_shape[2],
        staffline_patches_shape[3] * staffline_patches_shape[4]
    ])
    distance_matrix = _squared_euclidean_distance_matrix(
        flattened_patches, flattened_centroids)

    # Take the k centroids with the lowest distance to each patch. Wrap the k
    # constant in a tf.identity, which tests can use to feed in another value.
    k_value = tf.identity(tf.constant(K_NEAREST_VALUE), name='k_nearest_value')
    nearest_centroid_inds = tf.nn.top_k(-distance_matrix, k=k_value)[1]
    # Get the label corresponding to each nearby centroids, and reshape the
    # labels back to the original shape.
    nearest_labels = tf.reshape(
        tf.gather(labels, tf.reshape(nearest_centroid_inds, [-1])),
        tf.shape(nearest_centroid_inds))
    # Make a histogram of counts for each glyph type in the nearest centroids,
    # for each row (patch).
    bins = tf.map_fn(lambda row: tf.bincount(row, minlength=NUM_GLYPHS),
                     tf.to_int32(nearest_labels))
    # Take the argmax of the histogram to get the top prediction. Discard glyph
    # type 1 (NONE) for now.
    mode_out_of_k = tf.argmax(
        bins[:, musicscore_pb2.Glyph.NONE + 1:], axis=1) + 2
    # Force predictions to NONE only if all k nearby centroids were NONE.
    # Otherwise, the non-NONE nearby centroids will contribute to the
    # prediction.
    mode_out_of_k = tf.where(
        tf.equal(bins[:, musicscore_pb2.Glyph.NONE], k_value),
        tf.fill(
            tf.shape(mode_out_of_k), tf.to_int64(musicscore_pb2.Glyph.NONE)),
        mode_out_of_k)
    predictions = tf.reshape(mode_out_of_k, staffline_patches_shape[:3])

    # Pad the output.
    predictions_width = tf.shape(predictions)[-1]
    pad_before = (width - predictions_width) // 2
    pad_shape_before = tf.concat([staffline_patches_shape[:2], [pad_before]],
                                 axis=0)
    pad_shape_after = tf.concat(
        [staffline_patches_shape[:2], [width - predictions_width - pad_before]],
        axis=0)
    self.output = tf.concat(
        [
            # NONE has value 1.
            tf.ones(pad_shape_before, tf.int64),
            predictions,
            tf.ones(pad_shape_after, tf.int64),
        ],
        axis=-1)

  @property
  def staffline_predictions(self):
    return self.output


def _squared_euclidean_distance_matrix(a, b):
  # Trick for computing the squared Euclidean distance matrix.
  # Entry (i, j) = a[i].sum() + b[j].sum() - 2 * (a[i] * b[j]).sum()
  #              = sum_k (a[i, k] + b[j, k] - 2 * a[i, k] * b[j, k])
  #              = sum_k (a[i, k] - b[j, k]) ** 2
  a_sum = tf.reshape(tf.reduce_sum(a, axis=1), [-1, 1])  # column vector
  b_sum = tf.reshape(tf.reduce_sum(b, axis=1), [1, -1])  # row vector

  return a_sum + b_sum - 2 * tf.matmul(a, b, transpose_b=True)
