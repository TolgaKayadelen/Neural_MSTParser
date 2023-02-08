package(default_visibility = ["//visibility:public"])

# Binaries


py_binary(
    name = "biaffine_parser_main",
    srcs = ["biaffine_parser_main.py"],
    deps = [
        ":biaffine_parser",
        ":load_models",
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
    "//tagset:reader",
    "//eval:evaluate_v2"
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
  name = "label_first_parser_v2",
  srcs = ["label_first_parser_v2.py"],
  deps = [
    ":base_parser_v2",
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
  name = "base_parser_v2",
  srcs = ["base_parser_v2.py"],
  deps = [
      ":nn_parser_lib",
  ],
)

