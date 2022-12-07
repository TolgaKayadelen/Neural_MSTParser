package(default_visibility = ["//visibility:public"])

py_binary(
	name = "bilstm_labeler_main",
	srcs = ["bilstm_labeler_main.py"],
    deps = [
        ":bilstm_labeler",
        "//parser/utils:load_models",
  ],
)

py_binary(
	name = "bilstm_labeler_coarse_dep",
	srcs = ["bilstm_labeler_coarse_dep.py"],
    deps = [
        "//parser:load_models",
        "//parser:nn_parser_lib",
        "//util/nn:label_converter",
  ],
)

py_binary(
	name = "bilstm_labeler_fine_dep",
	srcs = ["bilstm_labeler_fine_dep.py"],
    deps = [
        "//parser/utils:load_models",
        "//parser:nn_parser_lib",
        "//util/nn:label_converter",
        "//tagset:reader"
  ],
)

py_binary(
	name = "bilstm_labeler_binary",
	srcs = ["bilstm_labeler_binary.py"],
    deps = [
        "//parser:base_parser",
        "//parser:nn_parser_lib",
        "//parser/utils:load_models",
  ],
)

py_binary(
	name = "bilstm_labeler_binary_softmax",
	srcs = ["bilstm_labeler_binary_softmax.py"],
    deps = [
        "//parser:base_parser",
        "//parser:nn_parser_lib",
        "//parser/utils:load_models",
  ],
)

# Libraries
py_library(
	name = "bilstm_labeler",
	srcs = ["bilstm_labeler.py"],
    deps = [
        "//parser:base_parser",
        "//parser:nn_parser_lib",
        "//input:embeddor",
  ],
)