# Description:
# OMR evaluation and music score similarity metrics.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "evaluation",
    deps = [
        ":evaluator",
        ":musicxml",
    ],
)

py_binary(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":musicxml",
        "//moonlight:engine",
        "//moonlight/conversions",
        "//moonlight/protobuf:protobuf_py_pb2",
        "@com_google_protobuf//:protobuf_python",
        # tensorflow dep
    ],
)

py_test(
    name = "evaluator_endtoend_test",
    srcs = ["evaluator_endtoend_test.py"],
    data = [
        "//moonlight/testdata:images",
        "//moonlight/testdata:musicxml",
    ],
    srcs_version = "PY2AND3",
    size = "large",
    deps = [
        ":evaluator",
        "//moonlight/protobuf:protobuf_py_pb2",
        # tensorflow dep
    ],
)

py_library(
    name = "musicxml",
    srcs = ["musicxml.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/music:constants",
        # lxml dep
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
