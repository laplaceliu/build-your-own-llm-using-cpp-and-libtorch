# chapters/chapter06_finetuning/CMakeLists.txt
# ---------------------------------------------------------------------------
# 第 6 章：针对分类的微调
#
# 依赖：
#   - 第 2 章 tokenizer（bpe_tokenizer.h/.cpp + data/ 词表）
#   - 第 3 章 attention.h
#   - 第 4 章 gpt.h
#   - 第 5 章 safetensors.h/.cpp（解析 GPT-2 预训练权重）
#
# 数据（首次运行前准备）：
#   cd chapters/chapter06_finetuning/data
#   curl -fsSL -o sms_spam_collection.zip \
#     https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
#   unzip -o sms_spam_collection.zip && mv SMSSpamCollection SMSSpamCollection.tsv
#
# 运行：
#   ./chapter06_finetuning [data_dir] [safetensors]
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapter06_finetuning
  src/main.cpp
  ../chapter02_text_data/src/bpe_tokenizer.cpp
  ../chapter05_pretraining/src/safetensors.cpp
)
target_include_directories(chapter06_finetuning PRIVATE
  include
  ../chapter02_text_data/include
  ../chapter03_attention/include
  ../chapter04_gpt/include
  ../chapter05_pretraining/include
)
target_compile_definitions(chapter06_finetuning PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  TOKENIZER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_text_data/data")
