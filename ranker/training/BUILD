package(default_visibility = ["//visibility:public"])

py_binary(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//ranker:ranker_preprocessor",
        "//ranker:data_extractor",
        "//parser/utils:load_models",
        "//parser/utils:layer_utils",
        "//input:embeddor",
        "//util:writer",
        "//util/nn:nn_utils"
    ],
)