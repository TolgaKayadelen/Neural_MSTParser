package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "main",
	srcs = ["main.py"],
	deps = [
		"//parser/perceptron:dependency_parser_lib",
		"//data/treebank:treebank_py_pb2",
    "//parser_main:evaluate",
		":parse",
    ":label",
		":train",
    ":parse_and_label",
	],
)

# Libraries
py_library(
	name = "parse",
	srcs = ["parse.py"],
	deps = [
		"//parser/perceptron:dependency_parser_lib",
		"//data/treebank:treebank_py_pb2",
	],
)

py_library(
  name = "label",
  srcs = ["label.py"],
  deps = [
    "//parser/perceptron:dependency_parser_lib",
    "//parser/perceptron:dependency_labeler",
    "//data/treebank:treebank_py_pb2",
  ],
)

py_library(
	name = "train",
	srcs = ["train.py"],
	deps = [
		"//parser/perceptron:dependency_parser_lib",
    "//parser/perceptron:dependency_labeler",
		"//data/treebank:treebank_py_pb2",
	],
)

py_library(
  name = "parse_and_label",
  srcs = ["parse_and_label.py"],
  deps = [
    ":parse",
    ":label",
    "//util:writer",
  ],
)