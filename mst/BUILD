package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "max_span_tree",
	srcs = ["max_span_tree.py"],
	deps = [
		"//data/treebank:sentence_py_pb2",
		"//util:common",
		"//util:reader",
	],
)

# Binaries.
py_binary(
	name = "max_span_tree_simple",
	srcs = ["max_span_tree_simple.py"],
)

# Tests.
py_test(
	name = "max_span_tree_test",
	size = "large",
	srcs = ["max_span_tree_test.py"],
	data = [
		"//data/testdata:cycles",
		"//data/testdata:connections",
		"//data/testdata:msts",
		"//data/testdata:generic",
		"//data/testdata:tokens",
		"//data/testdata:chuliuedmonds",
	],
	deps = [
		":max_span_tree"
	],
)

# Tests.
py_test(
	name = "max_span_tree_simple_test",
	size = "small",
	srcs = ["max_span_tree_simple_test.py"],
	data = [
		"//data/testdata:cycles",
	],
	deps = [
		":max_span_tree_simple",
		"//data/treebank:sentence_py_pb2",
	],
)

