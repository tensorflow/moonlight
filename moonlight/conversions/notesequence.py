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
"""Converts an OMR `Score` to a `NoteSequence`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protobuf import music_pb2

from moonlight.protobuf import musicscore_pb2


def score_to_notesequence(score):
  """Score to NoteSequence conversion.

  Args:
    score: A `tensorflow.moonlight.Score` message.

  Returns:
    A `tensorflow.magenta.NoteSequence` message containing the notes in the
    score.
  """
  return music_pb2.NoteSequence(notes=list(_score_notes(score)))


def page_to_notesequence(page):
  return score_to_notesequence(musicscore_pb2.Score(page=[page]))


def _score_notes(score):
  for page in score.page:
    for system in page.system:
      for staff in system.staff:
        for glyph in staff.glyph:
          if glyph.HasField('note'):
            yield glyph.note
