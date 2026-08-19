# chapters/chapter02_text_data/CMakeLists.txt
# ---------------------------------------------------------------------------
# 第 2 章：处理文本数据
#
# 独立构建：
#   cmake -S chapters/chapter02_text_data \
#         -B chapters/chapter02_text_data/build \
#         -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
#   cmake --build chapters/chapter02_text_data/build -j
#
# 运行（data 目录由 DATA_DIR 宏内嵌，无需额外参数）：
#   ./chapters/chapter02_text_data/build/chapter02_text_data
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapter02_text_data
  src/main.cpp
  src/bpe_tokenizer.cpp
  # torchtext 官方 GPT2BPEEncoder（third_party/torchtext/，来自 pytorch/text）
  src/torchtext_gpt2_bpe.cpp
  third_party/torchtext/gpt2_bpe_tokenizer.cpp
  third_party/torchtext/regex.cpp
)
target_include_directories(chapter02_text_data PRIVATE
  include
  third_party/torchtext
)
target_link_libraries(chapter02_text_data PRIVATE re2)  # torchtext regex 依赖 RE2
target_compile_definitions(chapter02_text_data PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
