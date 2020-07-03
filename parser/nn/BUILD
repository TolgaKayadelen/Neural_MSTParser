package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "labeler",
	srcs = ["labeler.py"],
	deps = [
		"labeler_lib",
	],
)

# Libraries
py_library(
	name = "labeler_lib",
	srcs = ["labeler.py"],
	deps = [
		"//learner/nn:bilstm",
		"//util:reader",
    "//model:tags_and_labels_enum_py_pb2",
	],
	data = [
		"//data/testdata:parser",
    "//model:label_to_enum",
	],
)