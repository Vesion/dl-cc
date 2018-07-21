package(default_visibility = ["//visibility:public"])

cc_library(
  name = "nn",
  hdrs = ["nn.h"],
)

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    ":nn",
  ],
  data = [
    ":train_set.in",
  ],
)
