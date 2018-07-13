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
"""Constants for music theory in OMR."""

# The indices of the pitch classes in a major scale.
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

NUM_SEMITONES_PER_OCTAVE = 12

# These constants are coincidentally equal.
# The size of the perfect fifth interval.
NUM_SEMITONES_IN_PERFECT_FIFTH = 7
# The number of pitch classes present in a diatonic scale (e.g. the major scale)
NUM_NOTES_IN_DIATONIC_SCALE = 7

# The consecutive base notes of a key signature are each separated by a fifth,
# or 7 semitones.
CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
# CIRCLE_OF_FIFTHS is declared as a constant for clarity, but can be generated
# from:
# CIRCLE_OF_FIFTHS = [
#     (i * NUM_SEMITONES_IN_PERFECT_FIFTH) % NUM_SEMITONES_PER_OCTAVE
#     for i in range(NUM_SEMITONES_PER_OCTAVE)
# ]
