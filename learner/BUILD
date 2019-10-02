package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

#Protos.
py_proto_library(
	name = "featureset_py_pb2",
	srcs = ["featureset.proto"],
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
	name = "perceptron",
	srcs = ["perceptron.py"],
	deps = [
		":perceptron_lib",
	],
)

# Libraries.
py_library(
	name = "feature_extractor_lib",
	srcs = ["feature_extractor.py"],
  data = [
    ":features",
  ],
	deps = [
		"//data/treebank:sentence_py_pb2",
		"//learner:featureset_py_pb2",
		"//util:reader",
		"//util:common",
	],
)

py_library(
	name = "perceptron_lib",
	srcs = ["perceptron.py"],
	data = [
		"//data/testdata:features",
		"//data/testdata:perceptron",
	],
	deps = [
		"//learner:feature_extractor_lib",
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

py_test(
	name = "arc_perceptron_test",
	size = "medium",
	srcs = ["arc_perceptron_test.py"],
	deps = [
		":perceptron_lib",
	],
)

py_test(
  name = "label_perceptron_test",
  srcs = ["label_perceptron_test.py"],
  deps = [
    ":perceptron_lib",
  ],
)

# Data.
filegroup(
    name = "features",
    srcs = glob([
        "features/*.txt",
    ]),
)
