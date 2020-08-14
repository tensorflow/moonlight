# Description:
# Common logic for all model training.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "batches",
    srcs = ["batches.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl dep
    ],
)

py_test(
    name = "batches_test",
    srcs = ["batches_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":batches",
        # disable_tf2
        # absl dep
        # tensorflow dep
    ],
)

py_library(
    name = "glyph_patches",
    srcs = ["glyph_patches.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":batches",
        ":label_weights",
        # absl dep
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/util:memoize",
        # tensorflow dep
        # tensorflow.contrib.estimator py dep
        # tensorflow.contrib.image py dep
    ],
)

py_test(
    name = "glyph_patches_test",
    srcs = ["glyph_patches_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":glyph_patches",
        # disable_tf2
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "hyperparameters",
    srcs = ["hyperparameters.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep
        # tensorflow dep
    ],
)

py_test(
    name = "hyperparameters_test",
    srcs = ["hyperparameters_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hyperparameters",
        # disable_tf2
        # numpy dep
    ],
)

py_library(
    name = "label_weights",
    srcs = ["label_weights.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl dep
        "//moonlight/protobuf:protobuf_py_pb2",
        # six dep
        # tensorflow dep
    ],
)

py_test(
    name = "label_weights_test",
    srcs = ["label_weights_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":label_weights",
        # disable_tf2
        # tensorflow dep
    ],
)
