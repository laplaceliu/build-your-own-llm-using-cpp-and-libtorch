# chapters/chapter07_instruction_tuning/CMakeLists.txt
# ---------------------------------------------------------------------------
# 第 7 章：通过微调遵循人类指令
#
# 依赖：
#   - 第 2 章 tokenizer（bpe_tokenizer.h/.cpp + data/ 词表）
#   - 第 3 章 attention.h
#   - 第 4 章 gpt.h
#   - 第 5 章 safetensors.h/.cpp + training.h
#   - nlohmann-json（指令数据集 + Ollama 调用）
#
# 数据（首次运行前准备）：
#   cd chapters/chapter07_instruction_tuning/data
#   curl -fsSL -o instruction-data.json \
#     https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json
#   curl -fsSL -L -o gpt2-medium.safetensors https://hf-mirror.com/gpt2-medium/resolve/main/model.safetensors
#
# 运行：
#   ./chapter07_instruction_tuning [data_dir] [epochs]
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapter07_instruction_tuning
  src/main.cpp
  ../chapter02_text_data/src/bpe_tokenizer.cpp
  ../chapter05_pretraining/src/safetensors.cpp
)
target_include_directories(chapter07_instruction_tuning PRIVATE
  include
  ../chapter02_text_data/include
  ../chapter03_attention/include
  ../chapter04_gpt/include
  ../chapter05_pretraining/include
)
target_compile_definitions(chapter07_instruction_tuning PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  TOKENIZER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_text_data/data")
