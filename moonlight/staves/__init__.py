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
"""Staff detection.

Holds the staff detector classes that can be used as part of an OMR pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from moonlight.staves import hough
from moonlight.staves import projection

# Alias the staff detectors to access them directly from the staves module.
# pylint: disable=invalid-name
FilteredHoughStaffDetector = hough.FilteredHoughStaffDetector
ProjectionStaffDetector = projection.ProjectionStaffDetector

# The default staff detector that should be used in production.
StaffDetector = FilteredHoughStaffDetector
