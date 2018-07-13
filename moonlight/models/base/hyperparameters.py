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
"""Wrapper which saves hyperparameters to a model collection.

The hyperparameters will be carefully tuned, and should be included in the
exported saved model to ensure reproducibility.
"""
# TODO(ringw): Try to get a standardized mechanism for saving params into
# TensorFlow.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf


def estimator_with_saved_params(estimator, params):
  """Wraps an estimator with hyperparameters to be stored in the saved model.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    params: A dict of string to constant value (string, number, or NumPy array).

  Returns:
    A wrapped `tf.estimator.Estimator`. The model of the new estimator extends
    the wrapped model, and also includes the hyperparameters in a collection
    where they can be inspected later.

  Raises:
    ValueError: If a hyperparameter value is None.
  """
  # Validate parameters immediately, not in the model_fn.
  for name, value in six.iteritems(params):
    if value is None:
      raise ValueError('Hyperparameter cannot be None: {}'.format(name))

  # Estimator is mostly just a wrapper around the model_fn callable. Our wrapper
  # just needs a callable that adds all the params to a collection, and then
  # invokes the original callable.
  def model_fn(features, labels, mode, params, config):
    """Wraps the delegate estimator model_fn.

    Args:
      features: A dict of string to Tensor. Features to classify.
      labels: A Tensor with example labels, or None for prediction.
      mode: The mode string for the estimator.
      params: Passed through the newly constructed Estimator. These should be
        identical to the outer function's params.
      config: A TensorFlow estimator config object.

    Returns:
      An object holding the predictions, optimizer, etc.
    """
    with tf.name_scope('params'):
      for name, value in six.iteritems(params):
        tf.add_to_collection('params', tf.constant(name=name, value=value))

    # TODO(ringwalt): Some estimators may want the params. However, we currently
    # only use the DNNClassifier estimator, which does not use the params in its
    # model_fn. Therefore, we only pass the params into the model at all in
    # order to encode them here, and they are otherwise only used outside of the
    # model_fn when constructing the estimator.
    return estimator.model_fn(features, labels, mode, config)

  return tf.estimator.Estimator(
      model_fn, model_dir=estimator.model_dir, params=params)
