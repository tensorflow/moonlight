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
"""Fixes duplicate rests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from moonlight.glyphs import glyph_types


class FixRepeatedRests(object):

  def apply(self, page):
    """Remove duplicate rests of the same type."""
    for system in page.system:
      for staff in system.staff:
        to_remove = []
        last_rest = None
        for glyph in staff.glyph:
          if (last_rest and glyph_types.is_rest(glyph) and
              last_rest.type == glyph.type and
              glyph.x - last_rest.x < staff.staffline_distance):
            to_remove.append(glyph)
          last_rest = glyph

        for glyph in to_remove:
          staff.glyph.remove(glyph)

    return page
