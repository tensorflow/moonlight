"""More iterator utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random


def iter_sample(iterator, count, rand=None):
  """Performs reservoir sampling on an iterator.

  The output is a list. The entire iterator must be read in one shot to
  determine any output element, and `count` elements need to be stored in
  memory.

  Args:
    iterator: An iterator/generator.
    count: The number of elements to sample.
    rand: Optional random object which is already seeded.

  Returns:
    A list with length `count`, or the contents of `iterator` if smaller.
  """

  rand = rand or random.Random()
  result = []
  for index, elem in enumerate(iterator):
    # Fill the result with count elements.
    if index < count:
      result.append(elem)

    # Replace an existing element uniformly randomly, but the probability of
    # replacing any element is steadily decreasing.
    random_index = rand.randint(0, index)
    if random_index < count:
      result[random_index] = elem

  return result
