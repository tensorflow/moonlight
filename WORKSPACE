http_archive(
    name = "com_google_protobuf",
    sha256 = "13d3c15ebfad8c28bee203dd4a0f6e600d2a7d2243bac8b5d0e517466500fcae",
    strip_prefix = "protobuf-3.5.1",
    url = "https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-python-3.5.1.tar.gz",
)

new_http_archive(
    name = "six_archive",
    build_file = "six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
)

bind(
    name = "six",
    actual = "@six_archive//:six",
)

new_http_archive(
    name = "magenta",
    strip_prefix = "magenta-48a199085e303eeae7c36068f050696209b856bb/magenta",
    url = "https://github.com/tensorflow/magenta/archive/48a199085e303eeae7c36068f050696209b856bb.tar.gz",
    sha256 = "931fb7b57714d667db618b0c31db1444e44baab17865ad66b76fd24d9e20ad6d",
    build_file_content = "",
)
