# This package contains tools to create data input pipeline for the learners. 

package(default_visibility = ["//visibility:public"])

# Binaries
py_binary(
  name = "preprocessor",
  srcs = ["preprocessor.py"],
  deps = [
    "//util/nn:nn_utils",
    "//util:reader",
    "//util:writer",
  ],
)
