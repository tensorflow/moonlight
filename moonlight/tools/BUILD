package(default_visibility = ["//moonlight:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "gen_structure_test_case",
    srcs = ["gen_structure_test_case.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight:engine",
        # absl dep
    ],
)

py_binary(
    name = "export_kmeans_centroids",
    srcs = ["export_kmeans_centroids.py"],
    data = ["//moonlight/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        "//moonlight/glyphs:corpus",
        "//moonlight/glyphs:knn_model",
        # absl dep
        # tensorflow dep
    ],
)

py_test(
    name = "export_kmeans_centroids_test",
    srcs = ["export_kmeans_centroids_test.py"],
    srcs_version = "PY2AND3",
    # Original centroids were deleted from Piper, this test can no longer be run at head.
    # Will delete soon assuming we don't need it again.
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":export_kmeans_centroids",
        "//moonlight:engine",
        "//moonlight/glyphs:saved_classifier",
        # backports.tempfile dep
        # tensorflow dep
    ],
)
