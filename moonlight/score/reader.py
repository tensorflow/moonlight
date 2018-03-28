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
"""Reads Pages of glyphs and outputs a NoteSequence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from moonlight.protobuf import musicscore_pb2
from moonlight.score import measures
from moonlight.score import state
from moonlight.score.elements import clef

# The expected y position for clefs.
TREBLE_CLEF_EXPECTED_Y = -2
BASS_CLEF_EXPECTED_Y = 2

# 4 beats to a whole note.
REST_DURATIONS_ = {
    musicscore_pb2.Glyph.REST_QUARTER: 4 / 4,
    musicscore_pb2.Glyph.REST_EIGHTH: 4 / 8,
    musicscore_pb2.Glyph.REST_SIXTEENTH: 4 / 16,
}


class ScoreReader(object):
  """Reads a Score proto and interprets musical elements from the glyphs.

  Given a Page containing glyphs, holds global state for the entire score, and
  per-measure state (accidentals). Each glyph is added to the NoteSequence based
  on the current state.

  OMR is a work in progress. Voice detection is not yet implemented; the score
  is assumed to be monophonic.
  """

  def __init__(self):
    self.time = 0.0
    self.score_state = state.ScoreState()

  def __call__(self, score):
    """Reads a `tensorflow.moonlight.Score` message.

    Modifies the message in place to add detected musical elements.

    Args:
      score: A `tensorflow.moonlight.Score` message.

    Returns:
      The same Score object.
    """
    for page in score.page:
      self.read_page(page)
    # Modifies the score in place.
    return score

  def read_page(self, page):
    """Reads a `tensorflow.moonlight.Page` message.

    Modifies the page in place to add detected musical elements.

    Args:
      page: A `tensorflow.moonlight.Page` message.

    Returns:
      The same Page object.
    """
    for system in page.system:
      self.read_system(system)
    return page

  def read_system(self, system):
    self.score_state.num_staves(len(system.staff))
    system_measures = measures.Measures(system)
    for measure_num in xrange(system_measures.size()):
      for staff, staff_state in zip(system.staff, self.score_state.staves):
        for glyph in staff.glyph:
          if system_measures.get_measure(glyph) == measure_num:
            self._read_glyph(glyph, staff_state)
      self.score_state.add_measure()

  def _read_glyph(self, glyph, staff_state):
    ScoreReader.GLYPH_HANDLERS_[glyph.type](self, staff_state, glyph)

  def _read_clef(self, staff_state, glyph):
    """Reads a clef glyph.

    If the clef is at the expected y position, set the current clef.

    Args:
      staff_state: The state of the staff that the glyph is on.
      glyph: A glyph of type CLEF_TREBLE or CLEF_BASS.

    Raises:
      ValueError: If glyph is an unexpected type.
    """
    if glyph.type == musicscore_pb2.Glyph.CLEF_TREBLE:
      if glyph.y_position == TREBLE_CLEF_EXPECTED_Y:
        staff_state.set_clef(clef.TrebleClef())
    elif glyph.type == musicscore_pb2.Glyph.CLEF_BASS:
      if glyph.y_position == BASS_CLEF_EXPECTED_Y:
        staff_state.set_clef(clef.BassClef())
    else:
      raise ValueError('Unknown clef of type: ' +
                       musicscore_pb2.Glyph.Type.Name(glyph.type))

  def _read_note(self, staff_state, glyph):
    staff_state.measure_state.on_read_notehead()
    glyph.note.CopyFrom(staff_state.measure_state.get_note(glyph))

  def _read_rest(self, staff_state, glyph):
    staff_state.set_time(staff_state.get_time() + REST_DURATIONS_[glyph.type])

  def _read_accidental(self, staff_state, glyph):
    staff_state.measure_state.set_accidental(glyph.y_position, glyph.type)

  def _no_op_handler(self, glyph):
    pass

  GLYPH_HANDLERS_ = {
      musicscore_pb2.Glyph.NONE: _no_op_handler,
      musicscore_pb2.Glyph.CLEF_TREBLE: _read_clef,
      musicscore_pb2.Glyph.CLEF_BASS: _read_clef,
      musicscore_pb2.Glyph.NOTEHEAD_FILLED: _read_note,
      musicscore_pb2.Glyph.NOTEHEAD_EMPTY: _read_note,
      musicscore_pb2.Glyph.NOTEHEAD_WHOLE: _read_note,
      musicscore_pb2.Glyph.REST_QUARTER: _read_rest,
      musicscore_pb2.Glyph.REST_EIGHTH: _read_rest,
      musicscore_pb2.Glyph.REST_SIXTEENTH: _read_rest,
      musicscore_pb2.Glyph.FLAT: _read_accidental,
      musicscore_pb2.Glyph.SHARP: _read_accidental,
      musicscore_pb2.Glyph.NATURAL: _read_accidental,
  }
