load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")



# Python 2/3 compatibility.
http_archive(
    name = "six_archive",
    build_file = "@//third_party:six.BUILD",
    sha256 = "236bdbdce46e6e6a3d61a337c0f8b763ca1e8717c03b369e87a7ec7ce1319c0a",
    strip_prefix = "six-1.14.0",
    urls = [
        "https://pypi.python.org/packages/source/s/six/six-1.14.0.tar.gz",
    ],
)


# Google protocol buffers.
http_archive(
    name = "com_google_protobuf",
    sha256 = "a79d19dcdf9139fa4b81206e318e33d245c4c9da1ffed21c87288ed4380426f9",
    strip_prefix = "protobuf-3.11.4",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v3.11.4.tar.gz",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()


# Bazel Python rules.
http_archive(
    name = "rules_python",
    sha256 = "b5668cde8bb6e3515057ef465a35ad712214962f0b3a314e551204266c7be90c",
    strip_prefix = "rules_python-0.0.2",
    urls = [
        "https://github.com/bazelbuild/rules_python/releases/download/0.0.2/rules_python-0.0.2.tar.gz",
    ],
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# gRpc (only used for detecting and configuring local Python).
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "c0a6b40a222e51bea5c53090e9e65de46aee2d84c7fa7638f09cb68c3331b983",
    strip_prefix = "grpc-1.29.0",
    urls = [
        "https://github.com/grpc/grpc/archive/v1.29.0.tar.gz",
    ],
)

load(
    "@com_github_grpc_grpc//third_party/py:python_configure.bzl",
    "python_configure",
)

python_configure(name = "local_config_python")

bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)
