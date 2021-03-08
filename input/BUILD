# This package contains tools to create data input pipeline for the learners. 

package(default_visibility = ["//visibility:public"])

# Binaries
py_binary(
  name = "preprocessor",
  srcs = ["preprocessor.py"],
  deps = [
    "//tagset:reader",
    "//tagset/morphology:morph_tags",
    "//util/nn:nn_utils",
    "//util:reader",
  ],
)
