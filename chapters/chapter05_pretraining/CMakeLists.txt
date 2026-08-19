# chapters/chapter05_pretraining/CMakeLists.txt
# ---------------------------------------------------------------------------
# 第 5 章：在无标签数据上进行预训练
#
# 依赖：
#   - 第 2 章 tokenizer（bpe_tokenizer.h/.cpp + data/ 词表）
#   - 第 3 章 attention.h
#   - 第 4 章 gpt.h
#   - nlohmann-json（safetensors 解析）
#
# 运行：
#   ./chapter05_pretraining [data_dir] [epochs] [openai_safetensors]
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapter05_pretraining
  src/main.cpp
  src/safetensors.cpp
  ../chapter02_text_data/src/bpe_tokenizer.cpp
)
target_include_directories(chapter05_pretraining PRIVATE
  include
  ../chapter02_text_data/include
  ../chapter03_attention/include
  ../chapter04_gpt/include
)
target_compile_definitions(chapter05_pretraining PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_text_data/data")
