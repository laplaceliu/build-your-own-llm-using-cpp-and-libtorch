# chapters/chapter03_attention/CMakeLists.txt
# ---------------------------------------------------------------------------
# 第 3 章：编码注意力机制
#
# 独立构建：
#   cmake -S chapters/chapter03_attention \
#         -B chapters/chapter03_attention/build \
#         -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
#   cmake --build chapters/chapter03_attention/build -j
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapter03_attention src/main.cpp)
target_include_directories(chapter03_attention PRIVATE include)
