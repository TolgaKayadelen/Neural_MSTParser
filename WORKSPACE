load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# Google Protocol buffers.
# git_repository(
#    name = "com_google_protobuf",
#    remote = "https://github.com/google/protobuf.git",
#    commit = "fc5aa5d910eb0e58ea7dbd545f2bc0a61e80a323",
#)


http_archive(
	name = "com_google_protobuf",
	sha256 = "56b5d9e1ab2bf4f5736c4cfba9f4981fbc6976246721e7ded5602fbaee6d6869",
	strip_prefix = "protobuf-3.5.1.1",
	urls = ["https://github.com/google/protobuf/archive/v3.5.1.1.tar.gz"],
)

# Python 2/3 compatibility.
new_http_archive(
    name = "six_archive",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
    build_file = "bazel/six.BUILD",
)

bind(
    name = "six",
    actual = "@six_archive//:six",
)

# Bazel python rules.
git_repository(
    name = "io_bazel_rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    commit = "44711d8ef543f6232aec8445fb5adce9a04767f9",
)

load(
    "@io_bazel_rules_python//python:pip.bzl",
    "pip_repositories"
)
pip_repositories()
