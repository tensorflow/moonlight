package(default_visibility = ["//moonlight:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "gen_structure_test_case",
    srcs = ["gen_structure_test_case.py"],
    srcs_version = "PY2AND3",
    deps = [
        # disable_tf2
        # absl dep
        "//moonlight:engine",
    ],
)

py_binary(
    name = "export_kmeans_centroids",
    srcs = ["export_kmeans_centroids.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_kmeans_centroids_lib",
        # disable_tf2
    ],
)

py_library(
    name = "export_kmeans_centroids_lib",
    srcs = ["export_kmeans_centroids.py"],
    data = ["//moonlight/testdata:images"],
    srcs_version = "PY2AND3",
    deps = [
        # absl dep
        "//moonlight/glyphs:corpus",
        "//moonlight/glyphs:knn_model",
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
        ":export_kmeans_centroids_lib",
        # disable_tf2
        "//moonlight:engine",
        "//moonlight/glyphs:saved_classifier",
        # tensorflow dep
    ],
)
