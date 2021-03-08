package(default_visibility = ["//visibility:public"])

# Libraries
py_library(
	name = "reader",
	srcs = ["reader.py"],
	deps = [
    "//tagset/fine_pos:fine_tag_enum_py_pb2",
    "//tagset/coarse_pos:coarse_tag_enum_py_pb2",
    "//tagset/dep_labels:dep_label_enum_py_pb2",
    "//tagset/arg_str:semantic_role_enum_py_pb2",
    "//tagset/morphology:morph_tag_enum_py_pb2",
	],
)

# Tests.
py_test(
  name = "reader_test",
  srcs = ["reader_test.py"],
  deps = [
    ":reader",
  ],
)