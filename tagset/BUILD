package(default_visibility = ["//visibility:public"])

# Libraries
py_library(
	name = "reader",
	srcs = ["reader.py"],
	deps = [
	":en_tagset",
	":de_tagset",
	":fi_tagset",
	":zh_tagset",
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

py_library(
    name = "en_tagset",
    deps = [
        "//tagset/fine_pos/en:fine_tag_enum_py_pb2",
        "//tagset/coarse_pos/en:coarse_tag_enum_py_pb2",
        "//tagset/dep_labels/en:dep_label_enum_py_pb2",

    ]
)
py_library(
    name = "de_tagset",
    deps = [
        "//tagset/dep_labels/de:dep_label_enum_py_pb2",

    ]
)
py_library(
    name = "fi_tagset",
    deps = [
        "//tagset/dep_labels/fi:dep_label_enum_py_pb2",

    ]
)
py_library(
    name = "zh_tagset",
    deps = [
        "//tagset/dep_labels/zh:dep_label_enum_py_pb2",

    ]
)