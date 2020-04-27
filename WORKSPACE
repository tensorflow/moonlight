load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_google_protobuf",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v3.11.4.tar.gz",
    ],
    sha256 = "a79d19dcdf9139fa4b81206e318e33d245c4c9da1ffed21c87288ed4380426f9",
    strip_prefix = "protobuf-3.11.4",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

http_archive(
    name = "magenta",
    strip_prefix = "magenta-48a199085e303eeae7c36068f050696209b856bb/magenta",
    url = "https://github.com/tensorflow/magenta/archive/48a199085e303eeae7c36068f050696209b856bb.tar.gz",
    sha256 = "931fb7b57714d667db618b0c31db1444e44baab17865ad66b76fd24d9e20ad6d",
    build_file_content = "",
)

http_archive(
    name = "omr_regression_test_data",
    strip_prefix = "omr_regression_test_data_20180516",
    url = "https://github.com/tensorflow/moonlight/releases/download/v2018.05.16-data/omr_regression_test_data_20180516.tar.gz",
    sha256 = "b47577ee6b359c2cbbcdb8064c6bd463692c728c55ec5c0ab78139165ba8f35a",
    build_file_content = """

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "omr_regression_test_data",
    srcs = glob(["**/*.png"]),
)
""",
)
