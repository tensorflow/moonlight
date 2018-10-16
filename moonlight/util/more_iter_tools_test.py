"""Tests for more_iter_tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl.testing import absltest
from moonlight.util import more_iter_tools
import numpy as np
from six import moves


class MoreIterToolsTest(absltest.TestCase):

  def testSample_count_0(self):
    self.assertEqual([], more_iter_tools.iter_sample(moves.range(100), 0))

  def testSample_iter_empty(self):
    self.assertEqual([], more_iter_tools.iter_sample(moves.range(0), 10))

  def testSample_distribution(self):
    sample = more_iter_tools.iter_sample(
        moves.range(0, 100000), 9999, rand=random.Random(12345))
    self.assertEqual(9999, len(sample))

    # Create a histogram with 10 bins.
    bins = np.bincount([elem // 10000 for elem in sample])
    self.assertEqual(10, len(bins))

    # Samples should be distributed roughly uniformly into bins.
    expected_bin_count = 9999 // 10
    for bin_count in bins:
      self.assertTrue(
          np.allclose(bin_count, expected_bin_count, rtol=0.1),
          '{} within 10% of {}'.format(bin_count, expected_bin_count))


if __name__ == '__main__':
  absltest.main()
