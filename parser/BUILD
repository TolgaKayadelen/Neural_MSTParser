package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "dependency_parser",
	srcs = ["dependency_parser.py"],
	deps = [
		"dependency_parser_lib",
	],
)

#libraries
py_library(
	name = "decoder",
	srcs = ["decoder.py"],
	deps = [
		"//mst:max_span_tree",
		"//mst:max_span_tree_simple",
		"//util:common",
	],
)

py_library(
	name = "dependency_parser_lib",
	srcs = ["dependency_parser.py"],
	deps = [
		"//learner:perceptron_lib",
		"//parser:decoder",
	],
	data = [
		"//data/testdata:parser",
	],
)

# Tests
py_test(
	name = "dependency_parser_test",
	srcs = ["dependency_parser_test.py"],
	deps = [
		":dependency_parser_lib",
	],
)

py_test(
  name = "decoder_test",
  srcs = ["decoder_test.py"],
  deps = [
    ":decoder",
  ],
)
