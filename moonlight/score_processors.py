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
"""Processors that need to visit each page of the score in one pass.

These are intended for detecting musical elements, where musical context may
span staff systems and pages (e.g. the time signature). Musical elements (e.g.
notes) are added to the `Score` message directly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from moonlight.score import reader


def create_processors():
  yield reader.ScoreReader()


def process(score):
  """Processes a Score.

  Detects notes in the Score, and returns the Score in place.

  Args:
    score: A `Score` message.

  Returns:
    A `Score` message with `Note`s added to the `Glyph`s where applicable.
  """
  for processor in create_processors():
    score = processor(score)
  return score
