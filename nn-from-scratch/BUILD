package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "generate_trainset",
  srcs = ["generate_trainset.cc"],
)

cc_library(
  name = "nn",
  hdrs = ["nn.h"],
)

cc_binary(
  name = "main_debug",
  srcs = ["main.cc"],
  deps = [
    ":nn",
  ],
  data = [
    ":train_set.in",
  ],
  copts = [
    "-D __NN_DEBUG=1",
  ],
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
