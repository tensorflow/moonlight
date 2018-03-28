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
"""Tests for MusicXML output."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from protobuf import music_pb2
from moonlight.conversions import musicxml
from moonlight.protobuf import musicscore_pb2


class MusicXMLTest(absltest.TestCase):

  def testSmallScore(self):
    score = musicscore_pb2.Score(page=[
        musicscore_pb2.Page(system=[
            musicscore_pb2.StaffSystem(staff=[
                musicscore_pb2.Staff(glyph=[
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.NOTEHEAD_FILLED,
                        x=10,
                        y_position=0,
                        note=music_pb2.NoteSequence.Note(
                            start_time=0, end_time=1, pitch=71)),
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.NOTEHEAD_EMPTY,
                        x=110,
                        y_position=-6,
                        note=music_pb2.NoteSequence.Note(
                            start_time=1, end_time=2.5, pitch=61)),
                ]),
                musicscore_pb2.Staff(glyph=[
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.NOTEHEAD_WHOLE,
                        x=10,
                        y_position=2,
                        note=music_pb2.NoteSequence.Note(
                            start_time=0, end_time=4, pitch=50)),
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.NOTEHEAD_FILLED,
                        beam=[
                            musicscore_pb2.LineSegment(),
                            musicscore_pb2.LineSegment()
                        ],
                        x=110,
                        y_position=-4,
                        note=music_pb2.NoteSequence.Note(
                            start_time=4, end_time=4.25, pitch=60)),
                ]),
            ]),
        ]),
    ])
    self.assertEqual(
        """<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE score-partwise PUBLIC
    "-//Recordare//DTD MusicXML 3.0 Partwise//EN"
    "http://www.musicxml.org/dtds/partwise.dtd">

<score-partwise version="3.0">
  <part-list>
    <score-part id="P1">
      <part-name>Part 1</part-name>
    </score-part>
    <score-part id="P2">
      <part-name>Part 2</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1024</divisions>
        <time symbol="common">
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
      </attributes>
      <note>
        <voice>1</voice>
        <type>quarter</type>
        <duration>1024</duration>
        <pitch>
          <step>B</step>
          <alter>0</alter>
          <octave>4</octave>
        </pitch>
      </note>
      <note>
        <voice>1</voice>
        <type>half</type>
        <duration>1536</duration>
        <pitch>
          <step>C</step>
          <alter>1</alter>
          <octave>4</octave>
        </pitch>
      </note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes>
        <divisions>1024</divisions>
        <time symbol="common">
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
      </attributes>
      <note>
        <voice>1</voice>
        <type>whole</type>
        <duration>4096</duration>
        <pitch>
          <step>D</step>
          <alter>0</alter>
          <octave>3</octave>
        </pitch>
      </note>
      <note>
        <voice>1</voice>
        <type>16th</type>
        <duration>256</duration>
        <pitch>
          <step>C</step>
          <alter>0</alter>
          <octave>4</octave>
        </pitch>
      </note>
    </measure>
  </part>
</score-partwise>
""",
        musicxml.score_to_musicxml(score))


if __name__ == '__main__':
  absltest.main()
