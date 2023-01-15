package(default_visibility = ["//visibility:public"])

py_binary(
    name = "evaluate_parser",
    srcs = ["evaluate_parser.py"],
    deps = [
        "//parser/dep/lfp:label_first_parser",
        "//parser/utils:load_models",
        "//util:writer",
        "//util:reader",
        "//tagset:reader",
        "//data/treebank:treebank_py_pb2",
    ]
)