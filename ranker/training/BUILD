package(default_visibility = ["//visibility:public"])

py_binary(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//ranker/preprocessing:ranker_preprocessor",
        "//parser/utils:load_models",
        "//parser/utils:layer_utils",
        "//proto:ranker_data_py_pb2",
        "//ranker/eval:rank_eval",
        "//input:embeddor",
        "//util:writer",
        "//util:reader",
    ],
)