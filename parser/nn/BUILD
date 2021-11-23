package(default_visibility = ["//visibility:public"])

# Binaries
py_binary(
  name = "label_first_parser",
  srcs = ["label_first_parser.py"],
  deps = [
    ":base_parser",
    ":nn_parser_lib",
  ],
)

py_binary(
    name = "biaffine_parser",
    srcs = ["biaffine_parser.py"],
    deps = [
        ":base_parser",
        ":nn_parser_lib",
    ],
)

py_binary(
	name = "bilstm_labeler",
	srcs = ["bilstm_labeler.py"],
  deps = [
    ":base_parser",
    ":nn_parser_lib",
  ],
)

# Libraries
py_library(
  name = "nn_parser_lib",
  deps = [
    ":layer_utils",
    ":architectures",
    "//input:preprocessor",
    "//util/nn:nn_utils",
    "//util:writer",
    "//util:converter",
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
    name = "base_parser",
    srcs = ["base_parser.py"],
    deps = [
        ":nn_parser_lib",
    ],
)


py_library(
	name = "label_first_parser2",
	srcs = ["label_first_parser2.py"],
  deps = [
    ":nn_parser_lib",
  ],
)



py_library(
	name = "seq2seq_labeler",
	srcs = ["seq2seq_labeler.py"],
  deps = [
    ":nn_parser_lib",
  ],
)

py_library(
	name = "seq2lstm_labeler",
	srcs = ["seq2lstm_labeler.py"],
  deps = [
    ":nn_parser_lib",
  ],
)




# Experimental
py_library(
	name = "label_first_parser_joint_loss",
	srcs = ["label_first_parser_joint_loss.py"],
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


# Deprecated (to be deleted later)
py_library(
	name = "label_first_parser_deprecated",
	srcs = ["label_first_parser_deprecated.py"],
  deps = [
    ":nn_parser_lib",
  ],
)

py_library(
	name = "biaffine_parser_deprecated",
	srcs = ["biaffine_parser_deprecated.py"],
  deps = [
    ":nn_parser_lib",
  ],
)

py_library(
	name = "bilstm_labeler_deprecated",
	srcs = ["bilstm_labeler_deprecated.py"],
  deps = [
    ":nn_parser_lib",
  ],
)
