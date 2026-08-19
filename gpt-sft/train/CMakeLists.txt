# gpt-sft/train/CMakeLists.txt
add_executable(sft_train src/sft_train.cpp)
target_link_libraries(sft_train PRIVATE gpt_sft_core)
target_compile_options(sft_train PRIVATE ${TORCH_CXX_FLAGS})
target_compile_definitions(sft_train PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data"
  TOKENIZER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../chapters/chapter02_text_data/data")
