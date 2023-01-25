package(default_visibility = ["//visibility:public"])

# Binaries
py_binary(
    name = "label_first_parser_exp_main",
    srcs = ["label_first_parser_exp_main.py"],
    deps = [
        ":load_models",
        ":label_first_parser_exp",
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
    name = "seq_lstm_attn_main",
    srcs = ["seq_lstm_attn_main.py"],
    deps = [
        ":seq_lstm_attn",
        ":load_models",
    ]
)

py_binary(
    name = "biaffine_parser_main",
    srcs = ["biaffine_parser_main.py"],
    deps = [
        ":biaffine_parser",
        ":load_models",
    ],
)

py_binary(
    name = "sequential_parser_batch",
    srcs = ["sequential_parser_batch.py"],
    deps = [
        ":base_parser",
        ":nn_parser_lib",
    ],
)

py_binary(
    name = "sequential_parser_exp",
    srcs = ["sequential_parser_exp.py"],
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

py_binary(
    name = "combined_parser",
    srcs = ["combined_parser.py"],
    deps = [
        ":combined_model",
        ":combined_parser_memory",
    ],
)

py_binary(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
      ":base_parser",
      ":nn_parser_lib",
      ":combined_parser_memory",
      ":seq_lstm_attn",
      ":sequential_parser_exp",
      ":load_models",
      ":rl_env"
    ],
)

py_binary(
    name = "lfp_tag_ranker",
    srcs = ["lfp_tag_ranker.py"],
    deps = [
        ":load_models",
        ":nn_parser_lib",
        ":base_parser",
    ],
)

# Libraries
py_library(
  name = "nn_parser_lib",
  deps = [
    "//parser/utils:layer_utils",
    "//parser/utils:architectures",
    "//parser/utils:pos_tag_to_id",
    "//input:preprocessor",
    "//util/nn:nn_utils",
    "//util:writer",
    "//util:converter",
    "//tagset/dep_labels:dep_label_enum_py_pb2",
    "//tagset/fine_pos:fine_tag_enum_py_pb2",
    "//tagset/coarse_pos:coarse_tag_enum_py_pb2",
  ]
)

py_library(
    name = "rl_env",
    srcs = ["rl_env.py"],
    deps = [
        ":nn_parser_lib",
    ],
)

py_library(
    name = "combined_model",
    srcs = ["combined_model.py"],
    deps = [
        ":base_parser",
        "nn_parser_lib",
    ],
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
  name = "label_first_parser_exp",
  srcs = ["label_first_parser_exp.py"],
  deps = [
    ":base_parser",
    ":nn_parser_lib",
    ":bilstm_labeler_coarse_dep",
    ":bilstm_labeler_fine_dep",
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
	name = "seq_lstm_attn",
	srcs = ["seq_lstm_attn.py"],
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
  name = "base_parser",
  srcs = ["base_parser.py"],
  deps = [
      ":nn_parser_lib",
  ],
)

py_library(
    name = "combined_parser_memory",
    srcs = ["combined_parser_memory.py"],
    deps = [],
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
