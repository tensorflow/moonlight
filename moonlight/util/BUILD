# Description:
# General utilities for OMR.
package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "util",
    deps = [
        ":functional_ops",
        ":memoize",
        ":more_iter_tools",
        ":patches",
        ":run_length",
        ":segments",
    ],
)

py_library(
    name = "functional_ops",
    srcs = ["functional_ops.py"],
    srcs_version = "PY2AND3",
    deps = [],  # tensorflow dep
)

py_test(
    name = "functional_ops_test",
    srcs = ["functional_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":functional_ops",
        # tensorflow dep
    ],
)

py_library(
    name = "memoize",
    srcs = ["memoize.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "more_iter_tools",
    srcs = ["more_iter_tools.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "more_iter_tools_test",
    srcs = ["more_iter_tools_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":more_iter_tools",
        # absl dep
        # numpy dep
        # six dep
    ],
)

py_library(
    name = "patches",
    srcs = ["patches.py"],
    srcs_version = "PY2AND3",
    deps = [],  # tensorflow dep
)

py_test(
    name = "patches_test",
    srcs = ["patches_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":patches",
        # tensorflow dep
    ],
)

py_library(
    name = "run_length",
    srcs = ["run_length.py"],
    srcs_version = "PY2AND3",
    deps = [],  # tensorflow dep
)

py_test(
    name = "run_length_test",
    srcs = ["run_length_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":run_length",
        # tensorflow dep
    ],
)

py_library(
    name = "segments",
    srcs = ["segments.py"],
    srcs_version = "PY2AND3",
    deps = [
        # enum34 dep
        # tensorflow dep
    ],
)

py_test(
    name = "segments_test",
    size = "small",
    srcs = ["segments_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":segments",
        # absl/testing dep
        # numpy dep
        # tensorflow dep
    ],
)
