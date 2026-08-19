# chapters/chapter01_hello_torch/CMakeLists.txt
# ---------------------------------------------------------------------------
# 第 1 章：Hello LibTorch
#
# 本章同样可脱离顶层工程独立构建：
#   cmake -S chapters/chapter01_hello_torch \
#         -B chapters/chapter01_hello_torch/build \
#         -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
#   cmake --build chapters/chapter01_hello_torch/build -j
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapter01_hello_torch main.cpp)
