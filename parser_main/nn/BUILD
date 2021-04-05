package(default_visibility = ["//visibility:public"])


py_binary(
	name = "main",
	srcs = ["main.py"],
  deps = [
    "//parser/nn:nn_parser_lib",
    "//parser/nn:joint_biaffine_parser_copy",
    ":parse",
  ],
)

py_library(
	name = "parse",
	srcs = ["parse.py"],
  deps = [
    "//parser/nn:nn_parser_lib",
    "//util:common",
  ],
)
