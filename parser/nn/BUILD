package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "labeler",
	srcs = ["labeler.py"],
	deps = [
		"labeler_lib",
	],
)

# Binaries.
py_binary(
	name = "biaffine_parser",
	srcs = ["biaffine_parser.py"],
  deps = [
    "//input:preprocessor",
    "//util/nn:nn_utils",
    "//util:writer",
  ],
)

# Libraries
py_library(
	name = "labeler_lib",
	srcs = ["labeler.py"],
	deps = [
		"//learner/nn:bilstm",
		"//util:reader",
    "//util:common",
    "//tagset/fine_pos:fine_tag_enum_py_pb2",
    "//tagset/coarse_pos:coarse_tag_enum_py_pb2",
    "//tagset/dep_labels:dep_label_enum_py_pb2",
    "//tagset/arg_str:semantic_role_enum_py_pb2",
    "//tagset:reader",
	],
	data = [
		"//data/testdata:parser",
	],
)

# Tests.
py_test(
  name = "labeler_test",
  srcs = ["labeler_test.py"],
  deps = [
    ":labeler",
  ],
)