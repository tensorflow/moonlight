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
"""Configures the Apache Beam runner from the command line in pipelines.

Command-line flags for particular runners can be added here later, if necessary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import apache_beam

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'runner', 'DirectRunner',
    'The class name of the Apache Beam runner to use in the pipeline.')


def create_pipeline(**kwargs):
  return apache_beam.Pipeline(FLAGS.runner, **kwargs)
