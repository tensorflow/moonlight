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
"""Simple memoizer for a function/method/property."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MemoizedFunction(object):
  """Decorates a function to be memoized.

  Caches all invocations of the function with unique arguments. The arguments
  must be hashable.

  Decorated functions are not threadsafe. This decorator is currently used for
  TensorFlow graph construction, which happens in a single thread.
  """

  def __init__(self, function):
    self._function = function
    self._results = {}

  def __call__(self, *args):
    """Calls the function to be memoized.

    Args:
      *args: The args to pass through to the function. Keyword arguments are not
        supported.

    Raises:
      TypeError if an argument is unhashable.

    Returns:
      The memoized return value of the wrapped function. The return value will
      be computed exactly once for each unique argument tuple.
    """
    if args in self._results:
      return self._results[args]
    self._results[args] = self._function(*args)
    return self._results[args]
