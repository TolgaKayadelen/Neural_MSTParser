package(default_visibility = ["//visibility:public"])

py_binary(
  name = "biaffine_parser_main",
  srcs = ["biaffine_parser_main.py"],
  deps = [
    "//parser/utils:load_models",
    ":biaffine_parser",
  ],
)

py_library(
  name = "biaffine_parser",
  srcs = ["biaffine_parser.py"],
  deps = [
    "//parser:base_parser",
    "//parser:nn_parser_lib",
  ],
)
