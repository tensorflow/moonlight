# Description:
# Generated training data for OMR.
package(default_visibility = ["//moonlight:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "generation",
    srcs = ["generation.py"],
    data = ["vexflow_generator.js"],
    srcs_version = "PY2AND3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
        # apache-beam dep
        "//moonlight:engine",
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/staves:staffline_distance",
        "//moonlight/staves:staffline_extractor",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "generation_test",
    srcs = ["generation_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":generation",
        # disable_tf2
        "//moonlight/staves:staffline_extractor",
        # tensorflow dep
    ],
)

py_library(
    name = "image_noise",
    srcs = ["image_noise.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep
        # tensorflow_addons.image dep
    ],
)

py_binary(
    name = "vexflow_generator_pipeline",
    srcs = ["vexflow_generator_pipeline.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":generation",
        ":image_noise",
        # disable_tf2
        # absl dep
        # apache-beam dep
        "//moonlight/pipeline:pipeline_flags",
        # tensorflow dep
    ],
)
