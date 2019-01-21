package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

# Protos.
py_proto_library(
	name = "sentence_py_pb2",
	srcs = ["sentence.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
)


# Binaries.
py_binary(
	name = "converter",
	srcs = ["converter.py"],
	deps = [
	"//data/treebank:sentence_py_pb2",
	"//util:reader",
	"//util:writer",
	],
)

