
# Binaries.
py_binary(
	name = "max_span_tree",
	srcs = ["max_span_tree.py"],
	deps = [
		"//data/treebank:sentence_py_pb2",
		"//util:reader",
	],
)


# Tests.
py_test(
	name = "max_span_tree_test",
	size = "small",
	srcs = ["max_span_tree_test.py"],
	data = [
	"//data/testdata:cycles",
	"//data/testdata:connections",
	"//data/testdata:msts",
	"//data/testdata:generic",
	"//data/testdata:tokens",
	],
	deps = [
		":max_span_tree"
	],
)
