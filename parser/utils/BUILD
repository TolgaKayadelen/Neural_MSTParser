package(default_visibility = ["//visibility:public"])

py_library(
  name = "architectures",
  srcs = ["architectures.py"],
  deps = [
    ":layer_utils",
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

# Tests
py_test(
  name = "layer_utils_test",
  srcs = ["layer_utils_test.py"],
  deps = [
    ":layer_utils",
  ],
)

py_library(
    name = "load_models",
    srcs = ["load_models.py"],
    deps = [
    "//parser:nn_parser_lib",
    "//parser/dep/lfp:label_first_parser",
  ],
)