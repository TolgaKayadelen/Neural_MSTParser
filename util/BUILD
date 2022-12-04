package(default_visibility = ["//visibility:public"])

# Libraries.
py_library(
    name = "writer",
    srcs = ["writer.py"],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
	deps = [
	  "//data/treebank:sentence_py_pb2",
	  "//data/treebank:treebank_py_pb2",
      "//proto:metrics_py_pb2",
	],
)

# Binaries.
py_binary(
	name = "common",
	srcs = ["common.py"],
	deps = [
	  "//data/treebank:sentence_py_pb2",
	  ":reader"
	],
)

py_binary(
	name = "converter",
	srcs = ["converter.py"],
	deps = [
	  "//data/treebank:sentence_py_pb2",
	  "//data/treebank:treebank_py_pb2",
      "//util:common",
	  "//util:reader",
	  "//util:writer",
      "//util/nn:nn_utils",
	],
)

py_binary(
  name = "get_model_info",
  srcs = ["get_model_info.py"],
  deps = [
    "//util:common",
  ],
)

py_binary(
  name = "model",
  srcs = ["model.py"],
  deps = [
    "//util:common",
    "//learner:featureset_py_pb2",
  ],
)

py_binary(
  name = "get_tags_and_labels",
  srcs = ["get_tags_and_labels.py"],
  deps = [
    "//data/treebank:treebank_py_pb2",
    "//data/treebank:sentence_py_pb2",
    "//util:reader",
  ],
)

py_binary(
	name = "check_cooccuring_sent",
	srcs = ["check_cooccuring_sent.py"],
	deps = [
	  "//parser/perceptron:dependency_parser_lib",
	  "//data/treebank:treebank_py_pb2",
	],
)

py_binary(
    name = "morph_label_cooccurence",
    srcs = ["morph_label_cooccurence.py"],
    deps = [
        ":reader",
    ]
)

# Tests.
py_test(
	name = "converter_test",
	size = "small",
	srcs = ["converter_test.py"],
	data = [
	  "//data/testdata:converter",
	],
	deps = [
	  ":converter"
	],
)

py_test(
	name = "common_test",
	size = "small",
	srcs = ["common_test.py"],
	data = [
	  "//data/testdata:common",
	],
	deps = [
	  ":common"
	],
)
