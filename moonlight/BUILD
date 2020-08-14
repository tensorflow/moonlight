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
        # disable_tf2
        "@com_google_protobuf//:protobuf_python",
        # absl dep
        "//moonlight/conversions",
        "//moonlight/glyphs:saved_classifier_fn",
        # tensorflow dep
    ],
)

py_test(
    name = "omr_endtoend_test",
    size = "large",
    srcs = ["omr_endtoend_test.py"],
    data = ["//moonlight/testdata:images"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":engine",
        # disable_tf2
        # pillow dep
        # absl/testing dep
        # librosa dep
        # lxml dep
        "//moonlight/conversions",
        "@magenta//protobuf:music_py_pb2",
        # numpy dep
        # tensorflow.python.platform dep
    ],
)

py_test(
    name = "omr_regression_test",
    size = "large",
    srcs = ["omr_regression_test.py"],
    args = ["--corpus_dir=../omr_regression_test_data"],
    data = ["@omr_regression_test_data"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":engine",
        # disable_tf2
        # absl/testing dep
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/score:reader",
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
