# gpt-sft/serve/CMakeLists.txt
# 推理服务：HTTP server（sft_serve）+ 命令行推理（sft_chat）
add_executable(sft_serve src/sft_serve.cpp)
target_link_libraries(sft_serve PRIVATE gpt_sft_core)
target_compile_options(sft_serve PRIVATE ${TORCH_CXX_FLAGS})
target_compile_definitions(sft_serve PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data"
  TOKENIZER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../chapters/chapter02_text_data/data")

add_executable(sft_chat src/sft_chat.cpp)
target_link_libraries(sft_chat PRIVATE gpt_sft_core)
target_compile_options(sft_chat PRIVATE ${TORCH_CXX_FLAGS})
target_compile_definitions(sft_chat PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data"
  TOKENIZER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../chapters/chapter02_text_data/data")
