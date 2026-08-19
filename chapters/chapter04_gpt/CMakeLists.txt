# chapters/chapter04_gpt/CMakeLists.txt
# ---------------------------------------------------------------------------
# 第 4 章：从头实现 GPT 模型进行文本生成
#
# 依赖：
#   - 第 2 章 tokenizer（bpe_tokenizer.h/.cpp + data/ 词表文件）
#   - 第 3 章 attention.h（ch3::MultiHeadAttention）
#
# 独立构建：
#   cmake -S chapters/chapter04_gpt \
#         -B chapters/chapter04_gpt/build \
#         -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
#   cmake --build chapters/chapter04_gpt/build -j
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapter04_gpt
  src/main.cpp
  ../chapter02_text_data/src/bpe_tokenizer.cpp
)
target_include_directories(chapter04_gpt PRIVATE
  include
  ../chapter02_text_data/include
  ../chapter03_attention/include
)
target_compile_definitions(chapter04_gpt PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_text_data/data")
