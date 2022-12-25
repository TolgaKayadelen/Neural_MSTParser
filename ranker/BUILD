package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

#Protos.
py_proto_library(
	name = "feature_py_pb2",
	srcs = ["feature.proto"],
	protoc = "@com_google_protobuf//:protoc",
	default_runtime = "@com_google_protobuf//:protobuf_python",
)

# Binaries.
py_binary(
	name = "feature_extractor",
	srcs = ["feature_extractor.py"],
	deps = [
		":feature_extractor_lib",
	],
)

py_binary(
    name = "data_extractor",
    srcs = ["data_extractor.py"],
    deps = [
        "//parser:nn_parser_lib",
        "//parser/dep/lfp:label_first_parser",
        "//parser/labeler/bilstm:bilstm_labeler",
        "//parser/utils:load_models",
        "//ranker:feature_extractor",
        "//proto:ranker_data_py_pb2",
        "//tagset/dep_labels:dep_label_enum_py_pb2",
        "//util:writer",
        "//util:reader",
    ],
)

py_binary(
    name = "ranker_preprocessor",
    srcs = ["ranker_preprocessor.py"],
    deps = [
        "//input:preprocessor",
        "//input:embeddor",
        "//util:reader",
        "//util:writer",
        "//util/nn:nn_utils",
    ]
)


# Libraries.
py_library(
	name = "feature_extractor_lib",
	srcs = ["feature_extractor.py"],
	deps = [
		"//data/treebank:sentence_py_pb2",
		":feature_py_pb2",
		"//util:reader",
		"//util:common",
	],
)

py_library(
	name = "perceptron_lib",
	srcs = ["perceptron.py"],
	deps = [
		":feature_extractor_lib",
		"//util:writer",
	],
)

# Tests.
py_test(
	name = "feature_extractor_test",
	size = "small",
	srcs = ["feature_extractor_test.py"],
	deps = [
		":feature_extractor_lib"
	],
)
