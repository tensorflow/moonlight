# Description:
# Glyph classification for OMR.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "glyphs",
    deps = [
        ":base",
        ":corpus",
        ":geometry",
        ":glyph_types",
        ":knn",
        ":knn_model",
        ":neural",
        ":note_dots",
        ":repeated",
        ":saved_classifier",
    ],
)

py_library(
    name = "corpus",
    srcs = ["corpus.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY2AND3",
    deps = [
        # enum34 dep
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "convolutional_test",
    srcs = ["convolutional_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":convolutional",
        ":testing",
        # disable_tf2
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
        # pandas dep
        # tensorflow dep
    ],
)

py_library(
    name = "knn",
    srcs = ["knn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convolutional",
        ":corpus",
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/util:patches",
        # tensorflow dep
    ],
)

py_library(
    name = "knn_model",
    srcs = ["knn_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "knn_test",
    srcs = ["knn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":knn",
        # disable_tf2
        "//moonlight/protobuf:protobuf_py_pb2",
        # pandas dep
        # tensorflow dep
    ],
)

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "glyph_types",
    srcs = ["glyph_types.py"],
    srcs_version = "PY2AND3",
    deps = ["//moonlight/protobuf:protobuf_py_pb2"],
)

py_library(
    name = "neural",
    srcs = ["neural.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/protobuf:protobuf_py_pb2",
        # tensorflow dep
    ],
)

py_test(
    name = "neural_test",
    size = "small",
    srcs = ["neural_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":neural",
        # disable_tf2
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "saved_classifier",
    srcs = ["saved_classifier.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convolutional",
        "//moonlight/staves:staffline_extractor",
        "//moonlight/util:patches",
        # tensorflow dep
        # tensorflow.contrib.graph_editor py dep
        # tensorflow.contrib.util py dep
    ],
)

py_test(
    name = "saved_classifier_test",
    srcs = ["saved_classifier_test.py"],
    data = ["//moonlight/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        ":saved_classifier",
        # disable_tf2
        "//moonlight:image",
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/structure",
        # tensorflow dep
    ],
)

py_library(
    name = "saved_classifier_fn",
    srcs = ["saved_classifier_fn.py"],
    data = ["//moonlight/data/glyphs_nn_model_20180808"],
    srcs_version = "PY2AND3",
    deps = [":saved_classifier"],
)

py_library(
    name = "note_dots",
    srcs = ["note_dots.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":geometry",
        ":glyph_types",
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/structure:components",
        # numpy dep
    ],
)

py_library(
    name = "repeated",
    srcs = ["repeated.py"],
    srcs_version = "PY2AND3",
    deps = ["//moonlight/glyphs:glyph_types"],
)

py_library(
    name = "testing",
    testonly = True,
    srcs = ["testing.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convolutional",
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
        # tensorflow dep
    ],
)
