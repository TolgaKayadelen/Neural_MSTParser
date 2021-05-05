package(default_visibility = ["//visibility:public"])

# Libraries
py_library(
  name = "nn_parser_lib",
  deps = [
    ":layer_utils",
    ":architectures",
    "//input:preprocessor",
    "//util/nn:nn_utils",
    "//util:writer",
  ]
)

py_library(
  name = "layer_utils",
  srcs = ["layer_utils.py"],
  deps = [
    "//input:embeddor",
    "//util/nn:nn_utils",
  ],
)

py_library(
  name = "architectures",
  srcs = ["architectures.py"],
  deps = [
    ":layer_utils",
  ],
)

py_library(
	name = "label_first_parser",
	srcs = ["label_first_parser.py"],
  deps = [
    ":nn_parser_lib",
  ],
)

py_library(
	name = "biaffine_parser",
	srcs = ["biaffine_parser.py"],
  deps = [
    ":nn_parser_lib",
  ],
)

# Tests
py_test(
  name = "layer_utils_test",
  srcs = ["layer_utils_test.py"],
  deps = [
    ":layer_utils",
  ],
)
