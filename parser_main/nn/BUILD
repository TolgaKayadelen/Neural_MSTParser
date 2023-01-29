package(default_visibility = ["//visibility:public"])

py_binary(
    name="ranker_parse_and_eval",
    srcs = ["ranker_parse_and_eval.py"],
    deps = [
        "//eval:evaluate",
        "//parser/utils:load_models",
        "//proto:ranker_data_py_pb2",
        "//ranker/preprocessing:feature_extractor",
        "//ranker/preprocessing:ranker_preprocessor",
        "//ranker/training:model",
        "//tagset/dep_labels:dep_label_enum_py_pb2",
        "//util:reader",
        "//util:writer",
    ]
)

