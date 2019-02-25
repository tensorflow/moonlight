# Description:
# Staff detection and removal routines.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "staves",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hough",
        ":projection",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/util:memoize",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "filter",
    srcs = ["filter.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/util:segments",
        # tensorflow dep
    ],
)

py_library(
    name = "hough",
    srcs = ["hough.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":filter",
        ":staffline_distance",
        "//moonlight/util:memoize",
        "//moonlight/vision:hough",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "projection",
    srcs = ["projection.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        "//moonlight/util:memoize",
        # numpy dep
        # scipy dep
        # tensorflow dep
    ],
)

py_test(
    name = "detectors_test",
    srcs = ["detectors_test.py"],
    data = ["//moonlight/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        ":staffline_distance",
        ":staves",
        ":testing",
        "//moonlight:image",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "staff_processor",
    srcs = ["staff_processor.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "staff_processor_test",
    srcs = ["staff_processor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":staff_processor",
        ":testing",
        # absl/testing dep
        "//moonlight:engine",
        "//moonlight/glyphs:testing",
        "//moonlight/protobuf:protobuf_py_pb2",
        "//moonlight/structure",
        # numpy dep
    ],
)

py_library(
    name = "staffline_extractor",
    srcs = ["staffline_extractor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":removal",
        ":staves",
        # enum34 dep
        "//moonlight:image",
        # six dep
        # tensorflow dep
    ],
)

py_test(
    name = "staffline_extractor_test",
    size = "medium",
    srcs = ["staffline_extractor_test.py"],
    data = ["//moonlight/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        ":staffline_extractor",
        ":staves",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "staffline_distance",
    srcs = ["staffline_distance.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/util:run_length",
        "//moonlight/util:segments",
        # tensorflow dep
    ],
)

py_test(
    name = "staffline_distance_test",
    srcs = ["staffline_distance_test.py"],
    data = ["//moonlight/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        ":staffline_distance",
        "//moonlight:image",
        # tensorflow dep
    ],
)

py_library(
    name = "removal",
    srcs = ["removal.py"],
    deps = [
        "//moonlight/util:memoize",
        "//moonlight/util:segments",
        # tensorflow dep
    ],
)

py_test(
    name = "removal_test",
    srcs = ["removal_test.py"],
    data = ["//moonlight/testdata:images"],
    deps = [
        ":removal",
        ":staffline_distance",
        ":staves",
        "//moonlight:image",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "testing",
    testonly = 1,
    srcs = ["testing.py"],
    srcs_version = "PY2AND3",
    deps = [":base"],
)
