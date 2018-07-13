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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from moonlight.models.base import hyperparameters
import numpy as np
import tensorflow as tf


class HyperparametersTest(tf.test.TestCase):

  def testSimpleModel(self):
    learning_rate = np.float32(0.123)
    params = {'learning_rate': learning_rate}
    estimator = hyperparameters.estimator_with_saved_params(
        tf.estimator.DNNClassifier(
            hidden_units=[10],
            feature_columns=[tf.feature_column.numeric_column('feature')]),
        params)
    with self.test_session():
      # Build the estimator model.
      estimator.model_fn(features={'feature': tf.placeholder(tf.float32)},
                         labels=tf.placeholder(tf.float32),
                         mode='TRAIN',
                         config=None)
      # We should be able to pull hyperparameters out of the TensorFlow graph.
      # The entire graph will also be written to the saved model in training.
      self.assertEqual(
          learning_rate,
          tf.get_default_graph().get_tensor_by_name('params/learning_rate:0')
          .eval())


if __name__ == '__main__':
  tf.test.main()
