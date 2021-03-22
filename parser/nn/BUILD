package(default_visibility = ["//visibility:public"])

# Binaries.
py_binary(
	name = "labeler",
	srcs = ["labeler.py"],
	deps = [
		"labeler_lib",
	],
)

py_binary(
	name = "biaffine_parser",
	srcs = ["biaffine_parser.py"],
  deps = [
    "//input:preprocessor",
    "//util/nn:nn_utils",
    "//util:writer",
  ],
)

py_binary(
	name = "joint_biaffine_parser",
	srcs = ["joint_biaffine_parser.py"],
  deps = [
    "//input:preprocessor",
    "//util/nn:nn_utils",
    "//util:writer",
  ],
)

py_binary(
	name = "joint_biaffine_parser_copy",
	srcs = ["joint_biaffine_parser_copy.py"],
  deps = [
    "//input:preprocessor",
    "//input:embeddor",
    "//util/nn:nn_utils",
    "//util:writer",
    ":model_builder",
  ],
)



# Libraries
py_library(
  name = "model_builder",
  srcs = ["model_builder.py"],
  deps = [
    "//input:embeddor",
    "//util/nn:nn_utils",
  ],
)

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

py_test(
  name = "model_builder_test",
  srcs = ["model_builder_test.py"],
  deps = [
    ":model_builder",
  ],
)