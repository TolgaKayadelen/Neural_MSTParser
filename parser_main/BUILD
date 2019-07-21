package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "main",
	srcs = ["main.py"],
	deps = [
		"//parser:dependency_parser_lib",
		"//data/treebank:treebank_py_pb2",
		"parse",
		"train",
	],
)


# Libraries
py_library(
	name = "parse",
	srcs = ["parse.py"],
	deps = [
		"//parser:dependency_parser_lib",
		"//data/treebank:treebank_py_pb2",
	],
)

py_library(
	name = "train",
	srcs = ["train.py"],
	deps = [
		"//parser:dependency_parser_lib",
		"//data/treebank:treebank_py_pb2",
	],

)