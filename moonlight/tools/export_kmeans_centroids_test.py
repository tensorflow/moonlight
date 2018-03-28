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
"""End to end test for exporting the KNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from backports import tempfile
import librosa
import tensorflow as tf

from moonlight import engine
from moonlight.glyphs import saved_classifier
from moonlight.tools import export_kmeans_centroids


class ExportKmeansCentroidsTest(tf.test.TestCase):

  def testEndToEnd(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with engine.get_included_labels_file() as centroids:
        export_dir = os.path.join(tmpdir, 'export')
        export_kmeans_centroids.run(centroids.name, export_dir)

      # Now load the saved model.
      omr = engine.OMREngine(
          glyph_classifier_fn=saved_classifier.SavedConvolutional1DClassifier.
          glyph_classifier_fn(export_dir))
      filename = os.path.join(tf.resource_loader.get_data_files_path(),
                              '../testdata/IMSLP00747-000.png')
      notes = omr.run(filename, output_notesequence=True)
      # TODO(ringw): Fix the extra note that is detected before the actual
      # first eighth note.
      self.assertEqual(librosa.note_to_midi('C4'), notes.notes[1].pitch)
      self.assertEqual(librosa.note_to_midi('D4'), notes.notes[2].pitch)
      self.assertEqual(librosa.note_to_midi('E4'), notes.notes[3].pitch)


if __name__ == '__main__':
  tf.test.main()
