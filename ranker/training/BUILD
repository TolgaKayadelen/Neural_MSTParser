package(default_visibility = ["//visibility:public"])

py_binary(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//ranker/preprocessing:ranker_preprocessor",
        "//ranker/preprocessing:datapoint_generator",
        "//parser/utils:load_models",
        "//parser/utils:layer_utils",
        "//proto:ranker_data_py_pb2",
        "//ranker/eval:rank_eval",
        "//input:embeddor",
        "//util:writer",
        "//util:reader",
        "//util/nn:nn_utils"
    ],
)