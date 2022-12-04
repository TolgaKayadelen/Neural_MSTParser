package(default_visibility = ["//visibility:public"])

py_binary(
  name = "label_first_parser_main",
  srcs = ["label_first_parser_main.py"],
  deps = [
    "//parser/utils:load_models",
    ":label_first_parser",
  ],
)

py_library(
  name = "label_first_parser",
  srcs = ["label_first_parser.py"],
  deps = [
    "//parser:base_parser",
    "//parser:nn_parser_lib",
  ],
)