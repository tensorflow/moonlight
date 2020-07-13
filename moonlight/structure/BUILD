# Description:
# Music score structure detection above the staff level.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "structure",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":beams",
        ":components",
        ":verticals",
        "//moonlight/staves",
        "//moonlight/staves:base",
        "//moonlight/staves:removal",
        # tensorflow dep
    ],
)

py_test(
    name = "structure_test",
    srcs = ["structure_test.py"],
    data = ["//moonlight/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        ":structure",
        # disable_tf2
        "//moonlight:image",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "barlines",
    srcs = ["barlines.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
    ],
)

py_test(
    name = "barlines_test",
    srcs = ["barlines_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":barlines",
        ":beams",
        ":components",
        ":structure",
        ":verticals",
        # disable_tf2
        # absl/testing dep
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/staves:base",
        # numpy dep
    ],
)

py_library(
    name = "beams",
    srcs = ["beams.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":components",
        "//moonlight/vision:morphology",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "beam_processor",
    srcs = ["beam_processor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":components",
        "//moonlight/glyphs:glyph_types",
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
    ],
)

py_library(
    name = "components",
    srcs = ["components.py"],
    srcs_version = "PY2AND3",
    deps = [
        # enum34 dep
        # tensorflow dep
        # tensorflow.contrib.image py dep
    ],
)

py_test(
    name = "components_test",
    srcs = ["components_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":components",
        # disable_tf2
        # tensorflow dep
    ],
)

py_library(
    name = "section_barlines",
    srcs = ["section_barlines.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":barlines",
        ":components",
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
    ],
)

py_library(
    name = "stems",
    srcs = ["stems.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/glyphs:glyph_types",
        "//moonlight/protobuf:protobuf_py_pb2",
        # numpy dep
    ],
)

py_test(
    name = "stems_test",
    srcs = ["stems_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":beams",
        ":components",
        ":stems",
        ":structure",
        ":verticals",
        # disable_tf2
        # absl/testing dep
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/staves:base",
        # numpy dep
    ],
)

py_library(
    name = "verticals",
    srcs = ["verticals.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/util:functional_ops",
        "//moonlight/util:memoize",
        "//moonlight/util:segments",
        "//moonlight/vision:images",
        "//moonlight/vision:morphology",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "verticals_test",
    srcs = ["verticals_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":verticals",
        # disable_tf2
        "//moonlight/staves:testing",
        # numpy dep
        # tensorflow dep
    ],
)
