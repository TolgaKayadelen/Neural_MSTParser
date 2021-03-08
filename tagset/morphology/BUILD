package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

# Binaries
py_binary(
	name = "morph_tags",
	srcs = ["morph_tags.py"],
  deps = [
    "//util:reader",
  ],
)

py_proto_library(
	name = "morph_tag_enum_py_pb2",
	srcs = ["morph_tag_enum.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
)