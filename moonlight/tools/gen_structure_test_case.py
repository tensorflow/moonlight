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
"""Prints assert statements about the number of systems, staves, and barlines.

After running the tool, please verify that the output is completely correct
before copying and pasting it into omr_regression_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re

from absl import app

from moonlight import engine


def main(argv):
  pages = argv[1:]
  assert pages, 'Pass one or more PNG files'
  omr = engine.OMREngine()
  for i, filename in enumerate(pages):
    escaped_filename = re.sub(r'([\'\\])', r'\\\0', filename)
    page = omr.run(filename).page[0]
    # TODO(ringw): Use a real templating system (e.g. jinja or mako).
    if i > 0:
      print('')
    print('  def test%s_structure(self):' % _sanitized_basename(filename))
    print('    page = engine.OMREngine().run(')
    print('        \'%s\').page[0]' % escaped_filename)
    print('    self.assertEqual(len(page.system), %d)' % len(page.system))
    for i, system in enumerate(page.system):
      print('')
      print('    self.assertEqual(len(page.system[%d].staff), %d)' %
            (i, len(system.staff)))
      print('    self.assertEqual(len(page.system[%d].bar), %d)' %
            (i, len(system.bar)))


def _sanitized_basename(filename):
  filename, unused_ext = os.path.splitext(os.path.basename(filename))
  return re.sub('[^A-z0-9]+', '_', filename)


if __name__ == '__main__':
  app.run(main)
