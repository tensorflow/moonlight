# Description:
# The music score state. Holds information such as the key signature and current time in the score,
# which is used for reading notes from the score.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "state",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":measure",
        ":staff",
        # six dep
    ],
)

py_library(
    name = "measure",
    srcs = ["measure.py"],
    srcs_version = "PY2AND3",
    deps = [
        # enum34 dep
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/score/elements:key_signature",
        "@magenta//protobuf:music_py_pb2",
    ],
)

py_library(
    name = "staff",
    srcs = ["staff.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":measure",
        "//moonlight/score/elements:clef",
    ],
)
