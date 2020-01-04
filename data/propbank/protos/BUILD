package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

# Protos.
py_proto_library(
	name = "frame_py_pb2",
	srcs = ["frame.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
)

py_proto_library(
	name = "propbank_py_pb2",
	srcs = ["propbank.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
	deps = [
		"//data/propbank/protos:frame_py_pb2",
	],
)
