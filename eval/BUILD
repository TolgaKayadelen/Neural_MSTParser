package(default_visibility = ["//visibility:public"])

py_library(
	name = "evaluate",
	srcs = ["evaluate.py"],
	deps = [
		"//parser/perceptron:dependency_parser_lib",
		"//data/treebank:treebank_py_pb2",
    "//proto:evaluation_py_pb2",
    "//tagset:reader"
	],
)

# Tests
py_test(
	name = "evaluate_test",
	srcs = ["evaluate_test.py"],
	deps = [
		":evaluate",
	],
	data = [
		"//data/testdata:parser_main",
	],
)
