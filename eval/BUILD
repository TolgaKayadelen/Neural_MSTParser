package(default_visibility = ["//visibility:public"])

py_binary(
	name = "evaluate",
	srcs = ["evaluate.py"],
	deps = [
	    "//util:reader",
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
