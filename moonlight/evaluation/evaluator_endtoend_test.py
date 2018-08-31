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
"""Runs OMR evaluation end to end and asserts on the evaluation metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

from moonlight.evaluation import evaluator
from moonlight.protobuf import groundtruth_pb2


class EvaluatorEndToEndTest(tf.test.TestCase):

  def testIncludedGroundTruth(self):
    ground_truth = groundtruth_pb2.GroundTruth(
        ground_truth_filename=os.path.join(
            tf.resource_loader.get_data_files_path(),
            '../testdata/IMSLP00747.golden.xml'),
        page_spec=[
            groundtruth_pb2.PageSpec(
                filename=os.path.join(tf.resource_loader.get_data_files_path(),
                                      '../testdata/IMSLP00747-000.png')),
        ])
    results = evaluator.Evaluator().evaluate(ground_truth)
    # Evaluation score is known to be greater than 0.65.
    self.assertGreater(results['overall_score']['total', ''], 0.65)


if __name__ == '__main__':
  tf.test.main()
