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
"""Tests for the clefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import librosa

from moonlight.score.elements import clef


class ClefTest(absltest.TestCase):

  def testTrebleClef(self):
    self.assertEqual(clef.TrebleClef().y_position_to_midi(-8),
                     librosa.note_to_midi('A3'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(-6),
                     librosa.note_to_midi('C4'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(0),
                     librosa.note_to_midi('B4'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(1),
                     librosa.note_to_midi('C5'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(3),
                     librosa.note_to_midi('E5'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(4),
                     librosa.note_to_midi('F5'))
    self.assertEqual(clef.TrebleClef().y_position_to_midi(14),
                     librosa.note_to_midi('B6'))

  def testBassClef(self):
    self.assertEqual(clef.BassClef().y_position_to_midi(-10),
                     librosa.note_to_midi('A1'))
    self.assertEqual(clef.BassClef().y_position_to_midi(-7),
                     librosa.note_to_midi('D2'))
    self.assertEqual(clef.BassClef().y_position_to_midi(-5),
                     librosa.note_to_midi('F2'))
    self.assertEqual(clef.BassClef().y_position_to_midi(-1),
                     librosa.note_to_midi('C3'))
    self.assertEqual(clef.BassClef().y_position_to_midi(0),
                     librosa.note_to_midi('D3'))
    self.assertEqual(clef.BassClef().y_position_to_midi(6),
                     librosa.note_to_midi('C4'))
    self.assertEqual(clef.BassClef().y_position_to_midi(8),
                     librosa.note_to_midi('E4'))


if __name__ == '__main__':
  absltest.main()
