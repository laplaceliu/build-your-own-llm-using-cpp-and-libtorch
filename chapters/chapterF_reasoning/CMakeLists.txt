# chapters/chapterF_reasoning/CMakeLists.txt
# ---------------------------------------------------------------------------
# 附录 F：理解推理大语言模型 —— CoT / 自一致性 / 推理时扩展
#
# 构建：
#   cmake -S chapters/chapterF_reasoning \
#         -B chapters/chapterF_reasoning/build \
#         -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
#   cmake --build chapters/chapterF_reasoning/build -j
# 运行：
#   ./chapters/chapterF_reasoning/build/chapterF_reasoning -n 30 -k 5 -e 0.4
# ---------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(libtorch)

add_libtorch_executable(chapterF_reasoning
  src/main.cpp
)
target_include_directories(chapterF_reasoning PRIVATE include)
target_compile_definitions(chapterF_reasoning PRIVATE
  DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../chapter02_text_data/data")
