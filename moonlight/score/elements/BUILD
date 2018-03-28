# Description:
# Score elements which encapsulate custom logic for score reading.

package(
    default_visibility = ["//moonlight:__subpackages__"],
)

py_library(
    name = "clef",
    srcs = ["clef.py"],
    deps = [
        "//moonlight/music:constants",
        "//moonlight/protobuf:protobuf_py_pb2",
        # librosa dep
    ],
)

py_test(
    name = "clef_test",
    srcs = ["clef_test.py"],
    deps = [
        ":clef",
        # absl/testing dep
        # librosa dep
    ],
)

py_library(
    name = "key_signature",
    srcs = ["key_signature.py"],
    deps = [
        ":clef",
        "//moonlight/music:constants",
        "//moonlight/protobuf:protobuf_py_pb2",
        # librosa dep
    ],
)

py_test(
    name = "key_signature_test",
    srcs = ["key_signature_test.py"],
    deps = [
        ":clef",
        ":key_signature",
        "//moonlight/protobuf:protobuf_py_pb2",
        # absl/testing dep
    ],
)
