package(default_visibility = ["//visibility:public"])

# Libraries.
py_library(
    name = "writer",
    srcs = ["writer.py"],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
	deps = [
		"//data/treebank:sentence_py_pb2",
		"//data/treebank:treebank_py_pb2",
    "//proto:metrics_py_pb2",
	],
)

# Binaries.
py_binary(
	name = "common",
	srcs = ["common.py"],
	deps = [
		"//data/treebank:sentence_py_pb2",
		"//learner:featureset_py_pb2",
		":reader"
	],
)

py_binary(
	name = "converter",
	srcs = ["converter.py"],
	deps = [
		"//data/treebank:sentence_py_pb2",
		"//data/treebank:treebank_py_pb2",
    "//util:common",
		"//util:reader",
		"//util:writer",
    "//util/nn:nn_utils",
	],
)

# Tests.
py_test(
	name = "converter_test",
	size = "small",
	srcs = ["converter_test.py"],
	data = [
	"//data/testdata:converter",
	],
	deps = [
		":converter"
	],
)

py_test(
	name = "common_test",
	size = "small",
	srcs = ["common_test.py"],
	data = [
	"//data/testdata:common",
	],
	deps = [
		":common"
	],
)
