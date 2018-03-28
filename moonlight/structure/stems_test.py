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
"""Tests for stem detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from moonlight import structure
from moonlight.protobuf import musicscore_pb2
from moonlight.staves import base as staves_base
from moonlight.structure import beams
from moonlight.structure import components
from moonlight.structure import stems as stems_module
from moonlight.structure import verticals

Point = musicscore_pb2.Point  # pylint: disable=invalid-name


class StemsTest(absltest.TestCase):

  def testDummy(self):
    # Create a single staff, and a single vertical which is the correct height
    # of a stem. The vertical has x = 20 and goes from
    struct = structure.Structure(
        staff_detector=staves_base.ComputedStaves(
            staves=[[[10, 50], [90, 50]]],
            staffline_distance=[12],
            staffline_thickness=2,
            staves_interpolated_y=[[50] * 100]),
        beams=beams.ComputedBeams(np.zeros((0, 2, 2))),
        verticals=verticals.ComputedVerticals(
            lines=[[[20, 38], [20, 38 + 12 * 4]]]),
        connected_components=components.ComputedComponents([]))
    stems = stems_module.Stems(struct)
    # Create a Page with Glyphs.
    input_page = musicscore_pb2.Page(system=[
        musicscore_pb2.StaffSystem(staff=[
            musicscore_pb2.Staff(
                staffline_distance=12,
                center_line=[
                    musicscore_pb2.Point(x=10, y=50),
                    musicscore_pb2.Point(x=90, y=50)
                ],
                glyph=[
                    # Cannot have a stem because it's a flat.
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.FLAT, x=15, y_position=-1),
                    # On the right side of the stem, the correct distance away.
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.NOTEHEAD_FILLED,
                        x=25,
                        y_position=-1),
                    # Too high for the stem.
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.NOTEHEAD_FILLED,
                        x=25,
                        y_position=4),
                    # Too far right from the stem.
                    musicscore_pb2.Glyph(
                        type=musicscore_pb2.Glyph.NOTEHEAD_FILLED,
                        x=35,
                        y_position=-1),
                ])
        ])
    ])
    page = stems.apply(input_page)
    self.assertFalse(page.system[0].staff[0].glyph[0].HasField("stem"))
    self.assertTrue(page.system[0].staff[0].glyph[1].HasField("stem"))
    self.assertEquals(page.system[0].staff[0].glyph[1].stem,
                      musicscore_pb2.LineSegment(
                          start=Point(x=20, y=38),
                          end=Point(x=20, y=38 + 12 * 4)))
    self.assertFalse(page.system[0].staff[0].glyph[2].HasField("stem"))
    self.assertFalse(page.system[0].staff[0].glyph[3].HasField("stem"))


if __name__ == "__main__":
  absltest.main()
