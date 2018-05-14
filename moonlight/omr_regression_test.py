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
"""Tests OMR with corpus scores from Placer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re

from absl.app import flags
from absl.testing import absltest

from moonlight import engine
from moonlight.protobuf import musicscore_pb2
from moonlight.score import measures
from moonlight.score import reader

IMSLP_FILENAME = re.compile('IMSLP([0-9]{5,})-[0-9]{3}.png')

flags.DEFINE_string('corpus_dir', 'corpus', 'Path to the extracted IMSLP pngs.')

FLAGS = flags.FLAGS


class OmrRegressionTest(absltest.TestCase):

  def testIMSLP01963_106_multipleStaffSizes(self):
    page = engine.OMREngine().run(
        _get_imslp_path('IMSLP01963-106.png')).page[0]
    self.assertEqual(len(page.system), 3)

    for system in page.system:
      self.assertEqual(len(system.staff), 4)
      self.assertEqual(system.staff[0].staffline_distance, 14)
      self.assertEqual(system.staff[1].staffline_distance, 14)
      self.assertEqual(system.staff[2].staffline_distance, 22)
      self.assertEqual(system.staff[3].staffline_distance, 22)

  def testIMSLP00823_000_structure(self):
    page = engine.OMREngine().run(_get_imslp_path('IMSLP00823-000.png')).page[0]
    self.assertEqual(len(page.system), 6)

    self.assertEqual(len(page.system[0].staff), 2)
    self.assertEqual(len(page.system[0].bar), 7)

    self.assertEqual(len(page.system[1].staff), 2)
    self.assertEqual(len(page.system[1].bar), 7)

    self.assertEqual(len(page.system[2].staff), 2)
    # TODO(ringw): Detect thick repeat barlines correctly.
    # page.system[2] should have 6 bars.

    self.assertEqual(len(page.system[3].staff), 2)
    self.assertEqual(len(page.system[3].bar), 6)

    self.assertEqual(len(page.system[4].staff), 2)
    # TODO(ringw): Fix. page.system[4] should have 6 bars.

    self.assertEqual(len(page.system[5].staff), 2)
    self.assertEqual(len(page.system[5].bar), 6)

  def testIMSLP00823_008_mergeStandardAndBeginRepeatBars(self):
    page = engine.OMREngine().run(_get_imslp_path('IMSLP00823-008.png')).page[0]
    self.assertEqual(len(page.system), 6)

    self.assertEqual(len(page.system[0].staff), 2)
    self.assertEqual(len(page.system[0].bar), 6)

    self.assertEqual(len(page.system[1].staff), 2)
    self.assertEqual(len(page.system[1].bar), 6)

    self.assertEqual(len(page.system[2].staff), 2)
    self.assertEqual(len(page.system[2].bar), 7)

    self.assertEqual(len(page.system[3].staff), 2)
    self.assertEqual(len(page.system[3].bar), 6)

    self.assertEqual(len(page.system[4].staff), 2)
    self.assertEqual(len(page.system[4].bar), 6)
    # TODO(ringw): Detect BEGIN_REPEAT_BAR here.
    self.assertEqual(page.system[4].bar[0].type,
                     musicscore_pb2.StaffSystem.Bar.END_BAR)
    self.assertEqual(page.system[4].bar[1].type,
                     musicscore_pb2.StaffSystem.Bar.STANDARD_BAR)

    self.assertEqual(len(page.system[5].staff), 2)
    self.assertEqual(len(page.system[5].bar), 7)

  def testIMSLP39661_keySignature_CSharpMinor(self):
    page = engine.OMREngine().run(_get_imslp_path('IMSLP39661-000.png')).page[0]
    score_reader = reader.ScoreReader()
    score_reader.read_system(page.system[0])
    treble_sig = score_reader.score_state.staves[0].get_key_signature()
    self.assertEqual(treble_sig.get_type(), musicscore_pb2.Glyph.SHARP)
    self.assertEqual(len(treble_sig), 4)
    bass_sig = score_reader.score_state.staves[1].get_key_signature()
    self.assertEqual(bass_sig.get_type(), musicscore_pb2.Glyph.SHARP)
    # TODO(ringw): Get glyphs detected correctly in the bass signature.
    # self.assertEqual(len(bass_sig), 4)

  def testIMSLP00023_015_doubleNoteDots(self):
    """Tests note dots in system[1].staff[1] of the image."""
    page = engine.OMREngine().run(_get_imslp_path('IMSLP00023-015.png')).page[0]
    self.assertEqual(len(page.system), 6)

    system = page.system[1]
    system_measures = measures.Measures(system)
    staff = system.staff[1]

    # All dotted notes in the first measure belong to one chord, and are
    # double-dotted.
    double_dotted_notes = [
        glyph for glyph in staff.glyph
        if system_measures.get_measure(glyph) == 0 and len(glyph.dot) == 2
    ]
    for note in double_dotted_notes:
      self.assertEqual(len(note.beam), 1)
      # Double-dotted eighth note duration.
      self.assertEqual(note.note.end_time - note.note.start_time,
                       .5 + .25 + .125)
    double_dotted_note_ys = [glyph.y_position for glyph in double_dotted_notes]
    self.assertIn(-6, double_dotted_note_ys)
    self.assertIn(-1, double_dotted_note_ys)
    self.assertIn(-3, double_dotted_note_ys)
    self.assertIn(3, double_dotted_note_ys)
    # TODO(ringw): Fix 3 dots detected at y position +4. The dots from y
    # position +3 are too close, and we should only consider a single row of
    # horizontally adjacent dots. For now, assert that there are no other notes
    # in the measure with 2 dots.
    self.assertTrue(
        set(double_dotted_note_ys).issubset([-6, -3, -1, 3, 4]),
        'No unexpected noteheads')

    # All dotted notes in the second measure belong to one chord, and are
    # single-dotted.
    single_dotted_notes = [
        glyph for glyph in staff.glyph
        if system_measures.get_measure(glyph) == 1 and len(glyph.dot) == 1
    ]
    for note in single_dotted_notes:
      self.assertEqual(len(note.beam), 1)
      # Single-dotted eighth note duration.
      self.assertEqual(note.note.end_time - note.note.start_time, .75)
    single_dotted_note_ys = [glyph.y_position for glyph in single_dotted_notes]
    self.assertIn(-5, single_dotted_note_ys)
    self.assertIn(-3, single_dotted_note_ys)
    # TODO(ringw): Fix the note at y position +2, incorrectly detected as
    # position +1.
    # TODO(ringw): Detect the dot for the note at y position +4.


def _get_imslp_path(filename):
  m = re.match(IMSLP_FILENAME, filename)
  bucket = int(m.group(1)) // 1000
  return os.path.join(FLAGS.corpus_dir, str(bucket), filename)


if __name__ == '__main__':
  absltest.main()
