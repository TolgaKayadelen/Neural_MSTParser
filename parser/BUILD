package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "dependency_parser",
	srcs = ["dependency_parser.py"],
	deps = [
		"//learner:perceptron_lib",
		"//parser:decoder",
	],
)


#libraries
py_library(
	name = "decoder",
	srcs = ["decoder.py"],
	deps = [
		"//mst:max_span_tree",
		"//util:common",
	],
)
