package(default_visibility = ["//visibility:public"])

# Libraries. TODO -- change this to library later.
py_binary(
	name = "nn_utils",
	srcs = ["nn_utils.py"],
	deps = [
	  "//data/treebank:sentence_py_pb2",
	  "//data/treebank:treebank_py_pb2",
	  "//util:reader",
	  "//util:writer",
      "//proto:metrics_py_pb2",
	    ],
    data = [
      "//embeddings:turkish_embeddings",
    ],
)

py_binary(
    name = "label_converter",
    srcs = ["label_converter.py"],
    deps = [
        "//util:writer",
        "//tagset/dep_labels:dep_label_enum_py_pb2",
    ],
)