# Description:
# Format conversions for OMR.
package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "conversions",
    srcs = ["__init__.py"],
    deps = [
        ":musicxml",
        ":notesequence",
    ],
)

py_library(
    name = "musicxml",
    srcs = ["musicxml.py"],
    deps = [
        # librosa dep
        # lxml dep
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/score:measures",
    ],
)

py_test(
    name = "musicxml_test",
    srcs = ["musicxml_test.py"],
    deps = [
        ":musicxml",
        "@magenta//protobuf:music_py_pb2",
        # absl/testing dep
        "//moonlight/protobuf:protobuf_py_pb2",
    ],
)

py_library(
    name = "notesequence",
    srcs = ["notesequence.py"],
    deps = [
        "@magenta//protobuf:music_py_pb2",
        "//moonlight/protobuf:protobuf_py_pb2",
    ],
)
