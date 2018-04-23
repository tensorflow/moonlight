#!/bin/bash
# Apache Beam only supports Python 2 :(
# Filter tests with tags = ["py2only"] for Python 3 (the TRAVIS_PYTHON_VERSION
# environment variable starts with a "3").
if [ "${TRAVIS_PYTHON_VERSION:0:1}" = 3 ]; then
  PYTHON_VERSION_FILTERS=--test_tag_filters=-py2only
fi

# End-to-end tests must be marked "large". They exceed Travis CI's 3GB RAM limit.
bazel test --test_output=errors --local_test_jobs=1 \
    --test_size_filters=small,medium $PYTHON_VERSION_FILTERS \
    //moonlight/...
