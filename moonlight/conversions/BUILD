# Description:
# Format conversions for OMR.
package(
    default_visibility = ["//moonlight:__subpackages__"],
)

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
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/score:measures",
        # librosa dep
        # lxml dep
    ],
)

py_test(
    name = "musicxml_test",
    srcs = ["musicxml_test.py"],
    deps = [
        ":musicxml",
        "//moonlight/protobuf:protobuf_py_pb2",
        "@magenta//protobuf:music_py_pb2",
        # absl/testing dep
    ],
)

py_library(
    name = "notesequence",
    srcs = ["notesequence.py"],
    deps = [
        "//moonlight/protobuf:protobuf_py_pb2",
        "@magenta//protobuf:music_py_pb2",
    ],
)
