cmake_minimum_required(VERSION 3.16)

project(build_your_own_llm LANGUAGES CXX)

# ---------------------------------------------------------------------------
# 编译器检查：LibTorch 2.x 要求 C++17，GCC 4.8 / 7 等老编译器无法使用。
# 若本机默认 g++ 过旧，请用新版本编译器重新配置，例如：
#   cmake -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
  message(FATAL_ERROR
    "默认编译器过旧（GCC ${CMAKE_CXX_COMPILER_VERSION}），LibTorch 2.x 需要 C++17。\n"
    "请用新版本编译器重新配置，例如：\n"
    "  cmake -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++-11\n"
    "  cmake --build build -j")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# 定位 LibTorch 并加载 add_libtorch_executable 帮助函数（见 cmake/libtorch.cmake）
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(libtorch)

# 按章节独立编排的 C++ 工程（每章可单独编译，也可由本文件一键全量构建）
add_subdirectory(chapters)

message(STATUS "== build-your-own-llm 配置完成 ==")
