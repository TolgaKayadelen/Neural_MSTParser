package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")


py_proto_library(
	name = "sentence_py_pb2",
	srcs = ["sentence.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
)

py_proto_library(
	name = "treebank_py_pb2",
	srcs = ["treebank.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
	deps = [
		"//proto:sentence_py_pb2",
	],
)

py_proto_library(
  name = "metrics_py_pb2",
  srcs = ["metrics.proto"],
  protoc = "@com_google_protobuf//:protoc",
  default_runtime =  "@com_google_protobuf//:protobuf_python",
)

py_proto_library(
	name = "error_analysis_py_pb2",
	srcs = ["error_analysis.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
)

py_proto_library(
    name = "ranker_data_py_pb2",
    srcs = ["ranker_data.proto"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf_python",
    deps = [
        "//ranker:feature_py_pb2",
    ]
)

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