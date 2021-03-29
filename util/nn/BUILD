package(default_visibility = ["//visibility:public"])

# Libraries. TODO -- change this to library later.
py_binary(
	name = "nn_utils",
	srcs = ["nn_utils.py"],
	deps = [
		"//data/treebank:sentence_py_pb2",
		"//data/treebank:treebank_py_pb2",
		"//util:reader",
    "//proto:metrics_py_pb2",
	],
  data = [
    "//embeddings:turkish_embeddings",
  ],
)