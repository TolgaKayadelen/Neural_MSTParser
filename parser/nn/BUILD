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
    "//tagset/fine_pos:fine_tag_enum_py_pb2",
    "//tagset/coarse_pos:coarse_tag_enum_py_pb2",
    "//tagset/dep_labels:dep_label_enum_py_pb2",
	],
	data = [
		"//data/testdata:parser",
	],
)