package(default_visibility = ["//visibility:public"])

# Binaries
py_binary(
  name = "rl_train",
  srcs = ["rl_train.py"],
  deps = [
    "//parser/nn:bilstm_labeler",
    "//parser/nn:nn_parser_lib",
    "//parser/nn:label_first_parser",
  ],
)