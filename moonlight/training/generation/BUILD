# Description:
# Generated training data for OMR.
package(default_visibility = ["//moonlight:__subpackages__"])

py_library(
    name = "generation",
    srcs = ["generation.py"],
    deps = [
        "//moonlight:engine",
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/staves:staffline_distance",
        "//moonlight/staves:staffline_extractor",
        "@com_google_protobuf//:protobuf_python",
        # apache-beam dep
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "generation_test",
    srcs = ["generation_test.py"],
    tags = ["manual"],
    deps = [
        ":generation",
        "//moonlight/staves:staffline_extractor",
        # tensorflow dep
    ],
)

py_library(
    name = "image_noise",
    srcs = ["image_noise.py"],
    srcs_version = "PY2AND3",
    deps = [],  # tensorflow dep
)
