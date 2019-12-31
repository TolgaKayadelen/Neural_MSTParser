package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "main",
	srcs = ["main.py"],
	deps = [
		"//parser:dependency_parser_lib",
		"//data/treebank:treebank_py_pb2",
		":parse",
    ":label",
		":train",
		":evaluate",
    ":parse_and_label",
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
  name = "label",
  srcs = ["label.py"],
  deps = [
    "//parser:dependency_parser_lib",
    "//parser:dependency_labeler",
    "//data/treebank:treebank_py_pb2",
  ],
)

py_library(
	name = "train",
	srcs = ["train.py"],
	deps = [
		"//parser:dependency_parser_lib",
    "//parser:dependency_labeler",
		"//data/treebank:treebank_py_pb2",
	],
)

py_library(
	name = "evaluate",
	srcs = ["evaluate.py"],
	deps = [
		"//parser:dependency_parser_lib",
		"//data/treebank:treebank_py_pb2",
    "//model:evaluation_py_pb2",
	],
	data = [
		"//model:label_to_enum",
	]
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
