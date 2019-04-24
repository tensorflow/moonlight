# Description:
# OMR evaluation and music score similarity metrics.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "evaluation",
    deps = [
        ":evaluator_lib",
        ":musicxml",
    ],
)

py_binary(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY2AND3",
    deps = [":evaluator_lib"],
)

py_library(
    name = "evaluator_lib",
    srcs = ["evaluator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":musicxml",
        "@com_google_protobuf//:protobuf_python",
        "//moonlight:engine",
        "//moonlight/conversions",
        "//moonlight/protobuf:protobuf_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "evaluator_endtoend_test",
    size = "large",
    srcs = ["evaluator_endtoend_test.py"],
    data = [
        "//moonlight/testdata:images",
        "//moonlight/testdata:musicxml",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":evaluator_lib",
        "//moonlight/protobuf:protobuf_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "musicxml",
    srcs = ["musicxml.py"],
    srcs_version = "PY2AND3",
    deps = [
        # lxml dep
        "//moonlight/music:constants",
        # pandas dep
        # six dep
    ],
)

py_test(
    name = "musicxml_test",
    srcs = ["musicxml_test.py"],
    data = ["//moonlight/testdata:musicxml"],
    srcs_version = "PY2AND3",
    deps = [
        ":musicxml",
        # absl/testing dep
        # lxml dep
        # pandas dep
        # tensorflow dep
    ],
)
