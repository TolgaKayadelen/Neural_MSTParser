package(default_visibility = ["//visibility:public"])

# Binaries
py_binary(
  name = "ac_train",
  srcs = ["ac_train.py"],
  deps = [
    "//parser/nn:load_models",
  ],
)

py_binary(
    name = "dqn",
    srcs = ["dqn.py"],
    deps = [
        "//parser/nn:load_models",
    ],
)
