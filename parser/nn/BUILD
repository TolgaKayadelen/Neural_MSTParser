package(default_visibility = ["//visibility:public"])

# Binaries

py_binary(
    name = "label_first_parser_main",
    srcs = ["label_first_parser_main.py"],
    deps = [
        ":load_models",
        ":label_first_parser",
    ],
)

py_binary(
	name = "bilstm_labeler_main",
	srcs = ["bilstm_labeler_main.py"],
    deps = [
        ":bilstm_labeler",
  ],
)

py_binary(
	name = "seq_lstm_labeler_main",
	srcs = ["seq_lstm_labeler_main.py"],
    deps = [
        ":seq_lstm_labeler",
  ],
)

py_binary(
	name = "seq_lstm_labeler_main_t",
	srcs = ["seq_lstm_labeler_main_t.py"],
    deps = [
        ":seq_lstm_labeler_t",
  ],
)

py_binary(
    name = "biaffine_parser_main",
    srcs = ["biaffine_parser_main.py"],
    deps = [
        ":biaffine_parser",
    ],
)

py_binary(
    name = "sequential_parser",
    srcs = ["sequential_parser.py"],
    deps = [
        ":base_parser",
        ":nn_parser_lib",
    ],
)

py_binary(
    name = "sequential_parser_base",
    srcs = ["sequential_parser_base.py"],
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
  name = "label_first_parser",
  srcs = ["label_first_parser.py"],
  deps = [
    ":base_parser",
    ":nn_parser_lib",
  ],
)

py_library(
	name = "bilstm_labeler",
	srcs = ["bilstm_labeler.py"],
    deps = [
        ":base_parser",
        ":nn_parser_lib",
  ],
)

py_library(
	name = "seq_lstm_labeler",
	srcs = ["seq_lstm_labeler.py"],
    deps = [
        ":base_parser",
        ":nn_parser_lib",
        "//tagset/dep_labels:dep_label_enum_py_pb2",
  ],
)

py_library(
	name = "seq_lstm_labeler_t",
	srcs = ["seq_lstm_labeler_t.py"],
    deps = [
        ":base_parser",
        ":nn_parser_lib",
        "//tagset/dep_labels:dep_label_enum_py_pb2",
  ],
)

py_library(
    name = "biaffine_parser",
    srcs = ["biaffine_parser.py"],
    deps = [
        ":base_parser",
        ":nn_parser_lib",
    ],
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
    name = "load_models",
    srcs = ["load_models.py"],
    deps = [
    ":nn_parser_lib",
    ":label_first_parser",
    ":bilstm_labeler",
  ],
)





# Deprecated
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


# Tests
py_test(
  name = "layer_utils_test",
  srcs = ["layer_utils_test.py"],
  deps = [
    ":layer_utils",
  ],
)