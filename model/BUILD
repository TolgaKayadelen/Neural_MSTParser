package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

# Protos.
py_proto_library(
	name = "tags_and_labels_enum_py_pb2",
	srcs = ["tag_and_labels_enum.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
)

# binaries.
py_binary(
  name = "get_model_info",
  srcs = ["get_model_info.py"],
  deps = [
    "//util:common",
  ],
)

py_binary(
  name = "get_postags",
  srcs = ["get_postags.py"],
  deps = [
    "//data/treebank:treebank_py_pb2",
    "//data/treebank:sentence_py_pb2",
    "//util:reader",
  ],
)
