# Description:
# Optical music recognition using TensorFlow.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

# The OMR engine. Entry point for running OMR.
py_library(
    name = "engine",
    srcs = ["engine.py"],
    data = ["//moonlight/data/knn_saved_model_20170514"],
    srcs_version = "PY2AND3",
    deps = [
        ":image",
        ":page_processors",
        ":score_processors",
        "//moonlight/conversions",
        "//moonlight/glyphs:saved_classifier_fn",
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/staves:base",
        "//moonlight/structure",
        "//moonlight/structure:beams",
        "//moonlight/structure:components",
        "//moonlight/structure:verticals",
        # numpy dep
        # six dep
        # tensorflow dep
    ],
)

# The omr CLI for running locally on a single score.
py_binary(
    name = "omr",
    srcs = ["omr.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":engine",
        "//moonlight/conversions",
        "//moonlight/glyphs:saved_classifier_fn",
        "@com_google_protobuf//:protobuf_python",
        # absl dep
        # tensorflow dep
    ],
)

py_test(
    name = "omr_endtoend_test",
    size = "medium",
    srcs = ["omr_endtoend_test.py"],
    data = ["//moonlight/testdata:images"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":engine",
        "//moonlight/conversions",
        "@magenta//protobuf:music_py_pb2",
        # pillow dep
        # absl/testing dep
        # librosa dep
        # lxml dep
        # numpy dep
        # tensorflow/python:platform dep
    ],
)

py_test(
    name = "omr_regression_test",
    size = "medium",
    srcs = ["omr_regression_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":engine",
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/score:reader",
        # absl/testing dep
    ],
)

py_library(
    name = "image",
    srcs = ["image.py"],
    srcs_version = "PY2AND3",
    deps = [],  # tensorflow dep
)

py_library(
    name = "page_processors",
    srcs = ["page_processors.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/glyphs:glyph_types",
        "//moonlight/glyphs:note_dots",
        "//moonlight/glyphs:repeated",
        "//moonlight/staves:staff_processor",
        "//moonlight/structure:barlines",
        "//moonlight/structure:beam_processor",
        "//moonlight/structure:section_barlines",
        "//moonlight/structure:stems",
    ],
)

py_library(
    name = "score_processors",
    srcs = ["score_processors.py"],
    srcs_version = "PY2AND3",
    deps = ["//moonlight/score:reader"],
)
