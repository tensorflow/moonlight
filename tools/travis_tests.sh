#!/bin/bash
# Print commands before running them.
set -x

# Apache Beam only supports Python 2 :(
# Filter tests with tags = ["py2only"] for Python 3 (the TRAVIS_PYTHON_VERSION
# environment variable starts with a "3").
if [ "${TRAVIS_PYTHON_VERSION:0:1}" = 3 ]; then
  PYTHON_VERSION_FILTERS=--test_tag_filters=-py2only
fi

# Test that we can build and import the "engine" module in the sandbox.
bazel build --incompatible_remove_native_http_archive=false //moonlight:omr
PYTHONPATH=sandbox python -m moonlight.engine

bazel test --incompatible_remove_native_http_archive=false \
    --test_output=errors --local_test_jobs=1 $PYTHON_VERSION_FILTERS \
    //moonlight/...
