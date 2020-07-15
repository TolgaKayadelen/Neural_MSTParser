package(default_visibility = ["//visibility:public"])

# Binaries
py_binary(
	name = "frame_converter",
	srcs = ["frame_converter.py"],
  deps = [
    "//util:writer",
    "//data/propbank/protos:frame_py_pb2",
    "//data/propbank/protos:propbank_py_pb2",
  ],
)