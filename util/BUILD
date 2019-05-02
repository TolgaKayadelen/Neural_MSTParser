package(default_visibility = ["//visibility:public"])

# Libraries
py_library(
    name = "writer",
    srcs = ["writer.py"],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
)

# Binaries
py_binary(
	name = "common",
	srcs = ["common.py"],
	deps = [
		"//data/treebank:sentence_py_pb2",
		":reader"
		],
)
