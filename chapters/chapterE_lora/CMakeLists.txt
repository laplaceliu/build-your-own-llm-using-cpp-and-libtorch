# chapters/chapterE_lora/CMakeLists.txt
# ---------------------------------------------------------------------------
# 附录 E：使用 LoRA 进行参数高效微调
#
# 依赖：ch2 (tokenizer) + ch3 (attention 参考) + ch4 (GPT / LayerNorm / GELU)
#
# 构建：
#   cmake -S chapters/chapterE_lora \
#         -B chapters/chapterE_lora/build \
#         -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
#   cmake --build chapters/chapterE_lora/build -j
# 运行：
#   ./chapters/chapterE_lora/build/chapterE_lora              # CPU
#   ./chapters/chapterE_lora/build/chapterE_lora --cuda       # GPU
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapterE_lora
  src/main.cpp
  ../chapter02_text_data/src/bpe_tokenizer.cpp
)
target_include_directories(chapterE_lora PRIVATE
  include
  ../chapter02_text_data/include
  ../chapter03_attention/include
  ../chapter04_gpt/include
)
target_compile_definitions(chapterE_lora PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_text_data/data")
