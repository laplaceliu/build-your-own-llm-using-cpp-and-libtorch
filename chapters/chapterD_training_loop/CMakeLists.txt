# chapters/chapterD_training_loop/CMakeLists.txt
# ---------------------------------------------------------------------------
# 附录 D：增强 LLM 的训练循环功能
#   D.1 学习率预热（线性预热）
#   D.2 余弦衰减
#   D.3 梯度裁剪
#   D.4 修改的训练函数
#
# 依赖：
#   - chapter02 (BPE tokenizer + data/)
#   - chapter04 (GPT model)
#   - chapter05 (dataloader + ch5::calc_loss_batch 等训练/评估辅助函数)
#
# 独立构建：
#   cmake -S chapters/chapterD_training_loop \
#         -B chapters/chapterD_training_loop/build \
#         -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
#   cmake --build chapters/chapterD_training_loop/build -j
#
# 运行：
#   ./chapters/chapterD_training_loop/build/chapterD_training_loop           # CPU
#   ./chapters/chapterD_training_loop/build/chapterD_training_loop --cuda    # GPU
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapterD_training_loop
  src/main.cpp
  ../chapter02_text_data/src/bpe_tokenizer.cpp
)
target_include_directories(chapterD_training_loop PRIVATE
  include
  ../chapter02_text_data/include
  ../chapter03_attention/include
  ../chapter04_gpt/include
  ../chapter05_pretraining/include
)
target_compile_definitions(chapterD_training_loop PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_text_data/data")
