package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

# Protos.
py_proto_library(
	name = "evaluation_py_pb2",
	srcs = ["evaluation.proto"],
  deps = [
    "//tagset/fine_pos:fine_tag_enum_py_pb2",
    "//tagset/coarse_pos:coarse_tag_enum_py_pb2",
    "//tagset/dep_labels:dep_label_enum_py_pb2",
      ],
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
  name = "model",
  srcs = ["model.py"],
  deps = [
  "//util:common",
  "//learner:featureset_py_pb2",
  ],
)

py_binary(
  name = "get_tags_and_labels",
  srcs = ["get_tags_and_labels.py"],
  deps = [
    "//data/treebank:treebank_py_pb2",
    "//data/treebank:sentence_py_pb2",
    "//util:reader",
  ],
)

# Data.
# (TODO) move this to tagset package, kill all dependencies to here.
filegroup(
    name = "label_to_enum",
    srcs = [":label_to_enum.tsv"],
)
