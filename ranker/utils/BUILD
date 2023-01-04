package(default_visibility = ["//visibility:public"])

py_binary(
    name = "ranker_utils",
    srcs = ["ranker_utils.py"],
    deps = [
        "//proto:ranker_data_py_pb2",
        "//ranker/training:model",
        "//ranker/preprocessing:ranker_preprocessor",
        "//parser/utils:load_models",
        "//util:writer",
        "//util:reader",
    ],
)