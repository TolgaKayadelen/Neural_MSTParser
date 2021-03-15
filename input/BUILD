# This package contains tools to create data input pipeline for the learners. 

package(default_visibility = ["//visibility:public"])

# Binaries
py_binary(
  name = "preprocessor",
  srcs = ["preprocessor.py"],
  deps = [
    "//tagset/morphology:morph_tags",
    "//tagset:reader",
    "//util/nn:nn_utils",
    "//util:reader",
    "//input:embeddor",
  ],
)

py_binary(
  name = "embeddor",
  srcs = ["embeddor.py"],
)

# Tests
# Tests
py_test(
	name = "preprocessor_test",
	srcs = ["preprocessor_test.py"],
	deps = [
		":preprocessor",
	],
)