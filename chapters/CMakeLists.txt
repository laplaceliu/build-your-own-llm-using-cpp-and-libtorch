# chapters/CMakeLists.txt
# ---------------------------------------------------------------------------
# 按章节独立编排：每一章都是一个独立子目录、独立的 CMake 工程、
# 独立的可执行程序。后续新增章节只需在下面追加一行 add_subdirectory。
# ---------------------------------------------------------------------------

add_subdirectory(chapter01_hello_torch)
add_subdirectory(chapter02_text_data)
add_subdirectory(chapter03_attention)
add_subdirectory(chapter04_gpt)
add_subdirectory(chapter05_pretraining)
