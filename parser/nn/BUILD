package(default_visibility = ["//visibility:public"])

py_binary(
	name = "biaffine_parser",
	srcs = ["biaffine_parser.py"],
  deps = [
    ":nn_parser_lib",
  ],
)

# Libraries
py_library(
  name = "nn_parser_lib",
  deps = [
    ":model_builder",
    "//input:preprocessor",
    "//util/nn:nn_utils",
    "//util:writer",
  ]
)

py_library(
  name = "model_builder",
  srcs = ["model_builder.py"],
  deps = [
    "//input:embeddor",
    "//util/nn:nn_utils",
  ],
)

py_library(
	name = "joint_biaffine_parser",
	srcs = ["joint_biaffine_parser.py"],
  deps = [
    ":nn_parser_lib",
  ],
)

py_library(
	name = "joint_biaffine_parser_copy",
	srcs = ["joint_biaffine_parser_copy.py"],
  deps = [
    ":nn_parser_lib",
  ],
)

# Tests
py_test(
  name = "model_builder_test",
  srcs = ["model_builder_test.py"],
  deps = [
    ":model_builder",
  ],
)
