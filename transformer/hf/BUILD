package(default_visibility = ["//visibility:public"])

py_binary(
    name = "bert_fine_tuner",
    srcs = ["bert_fine_tuner.py"],
    deps = [
        "//tagset:reader",
    ]
)

py_binary(
    name = "hf_dataset_creator",
    srcs = ["hf_dataset_creator.py"],
    deps = [
        "//util:reader",
        "//tagset:reader",
    ],
)
