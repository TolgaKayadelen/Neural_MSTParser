package(default_visibility = ["//visibility:public"])

# Pretrained layer weights which can be transferred across models are saved in this dir.

py_binary(
    name = "export_weights",
    srcs = ["export_weights.py"],
    deps = [
        "//parser/utils:load_models",
        "//input:preprocessor",
    ],
)
