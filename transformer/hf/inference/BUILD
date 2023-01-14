package(default_visibility = ["//visibility:public"])

py_binary(
    name = "predict",
    srcs = ["predict.py"],
    deps = [
        "//eval:evaluate",
        "//data/treebank:sentence_py_pb2",
        "//data/treebank:treebank_py_pb2",
        "//util:reader",
        "//util:writer",
        "//tagset:reader",
    ]
)