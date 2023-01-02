package(default_visibility = ["//visibility:public"])

py_binary(
  name = "label_first_parser_topk_main",
  srcs = ["label_first_parser_topk_main.py"],
  deps = [
    "//parser/utils:load_models",
    ":label_first_parser_topk",
  ],
)

py_library(
  name = "label_first_parser_topk",
  srcs = ["label_first_parser_topk.py"],
  deps = [
    "//parser:base_parser",
    "//parser:nn_parser_lib",
  ],
)