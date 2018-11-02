"""Extracts strips of the image for OCR which may contain a time signature.

TODO(ringwalt): DO NOT SUBMIT without a detailed description of time_signature.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from moonlight.staves import staffline_extractor


class TimeSignatureData(object):
  """Holds strips of the image that may contain time signature numerals."""

  def __init__(self, staff_remover):
    self.staff_remover = staff_remover
    detector = self.staff_remover.staff_detector
    extractor = staffline_extractor.StafflineExtractor(
        staff_remover.remove_staves, detector, num_sections=5,
        staffline_distance_multiple=2, target_height=32)
    # Skip the 3 sections in the middle that are not centered on a numeral.
    self.strips = extractor.extract_staves()[:, ::4]
    # Assert that each staff has 2 extracted strips.
    self.strips.set_shape((None, 2, None, None))

  @property
  def data(self):
    return [self.strips]


class ComputedTimeSignatureData(object):

  def __init__(self, strips):
    self.strips = strips

  @property
  def data(self):
    return [self.strips]
