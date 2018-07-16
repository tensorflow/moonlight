# Description:
# Score reading for OMR.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "score",
    deps = [
        ":measures",
        ":reader",
    ],
)

py_library(
    name = "measures",
    srcs = ["measures.py"],
    deps = ["//moonlight/protobuf:protobuf_py_pb2"],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
    deps = [
        ":measures",
        # absl dep
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/score/elements:clef",
        "//moonlight/score/state",
    ],
)

py_test(
    name = "reader_test",
    srcs = ["reader_test.py"],
    deps = [
        ":reader",
        "@magenta//protobuf:music_py_pb2",
        # absl/testing dep
        # librosa dep
        "//moonlight/conversions",
        "//moonlight/protobuf:protobuf_py_pb2",
    ],
)
