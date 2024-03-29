package(default_visibility = ["//visibility:public"])


py_binary(
    name = "partition_data",
    srcs = ["partition_data.py"],
    deps= [
        "//util:reader",
        "//util:writer",
        "//util:common",
    ]
)


py_binary(
    name = "global_stats",
    srcs = ["global_stats.py"],
    deps= [
        "//util:reader",
        "//util:writer",
        "//util:common",
    ]
)

py_test(
    name = "partition_test",
    srcs = ["partition_test.py"],
    deps = [
        ":partition_data",
        "//data/treebank:sentence_py_pb2"
    ]
)