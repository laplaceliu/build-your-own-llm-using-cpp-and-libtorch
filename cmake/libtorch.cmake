# cmake/libtorch.cmake
# ---------------------------------------------------------------------------
# 定位预编译 LibTorch，并定义 add_libtorch_executable 帮助函数。
#
# TORCH_ROOT 查找顺序：
#   1. 命令行参数：-DTORCH_ROOT=/path/to/libtorch
#   2. 环境变量：  $TORCH_ROOT
#   3. 默认路径：  /opt/libtorch
#
# 用法（在章节的 CMakeLists.txt 中）：
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
#   include(libtorch)
#   add_libtorch_executable(chapter01_hello_torch main.cpp)
# ---------------------------------------------------------------------------

set(TORCH_ROOT "" CACHE PATH "预编译 LibTorch 的根目录")

if(NOT TORCH_ROOT)
  if(DEFINED ENV{TORCH_ROOT} AND NOT "$ENV{TORCH_ROOT}" STREQUAL "")
    set(TORCH_ROOT "$ENV{TORCH_ROOT}" CACHE PATH "" FORCE)
  else()
    set(TORCH_ROOT "/opt/libtorch" CACHE PATH "" FORCE)
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_ROOT}")
find_package(Torch REQUIRED)

message(STATUS "LibTorch 版本: ${TORCH_VERSION}")
message(STATUS "Torch 库目录: ${TORCH_INSTALL_PREFIX}")

# 创建一个链接 LibTorch 的可执行目标。
# TORCH_LIBRARIES 中包含 imported target `torch`，其
# INTERFACE_INCLUDE_DIRECTORIES / INTERFACE_COMPILE_OPTIONS
# 会自动把头文件路径与 _GLIBCXX_USE_CXX11_ABI 等关键编译选项传给本目标。
function(add_libtorch_executable target)
  add_executable(${target} ${ARGN})
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  # 显式追加 torch 的 include 目录与编译选项，兼容 IDE / 子 cmake 工程
  if(DEFINED TORCH_INCLUDE_DIRS)
    target_include_directories(${target} PRIVATE ${TORCH_INCLUDE_DIRS})
  endif()
  if(DEFINED TORCH_CXX_FLAGS)
    target_compile_options(${target} PRIVATE ${TORCH_CXX_FLAGS})
  endif()
  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
endfunction()
