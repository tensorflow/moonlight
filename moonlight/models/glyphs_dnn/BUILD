# Description:
# Glyph patches DNN classifier.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl dep
        "//moonlight/models/base:glyph_patches",
        "//moonlight/models/base:hyperparameters",
        "//moonlight/protobuf:protobuf_py_pb2",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        # disable_tf2
        # absl dep
        "//moonlight/models/base:glyph_patches",
    ],
)
