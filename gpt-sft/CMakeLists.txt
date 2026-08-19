# gpt-sft/CMakeLists.txt
# 独立项目：GPT-2 指令微调（训练 + 推理服务分离）
#
# 训练：gpt-sft/train/sft_train    （指令微调，输出 .pth）
# 推理：gpt-sft/serve/sft_serve    （HTTP 服务）
#       gpt-sft/serve/sft_chat     （命令行推理）
cmake_minimum_required(VERSION 3.16)
project(gpt_sft LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
  message(FATAL_ERROR "需要支持 C++17 的编译器（建议 GCC 11）")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# LibTorch 定位（TORCH_ROOT 或默认 /opt/libtorch）
set(TORCH_ROOT "" CACHE PATH "预编译 LibTorch 根目录")
if(NOT TORCH_ROOT)
  if(DEFINED ENV{TORCH_ROOT} AND NOT "$ENV{TORCH_ROOT}" STREQUAL "")
    set(TORCH_ROOT "$ENV{TORCH_ROOT}")
  else()
    set(TORCH_ROOT "/opt/libtorch")
  endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_ROOT}")
find_package(Torch REQUIRED)
message(STATUS "LibTorch: ${TORCH_VERSION} @ ${TORCH_INSTALL_PREFIX}")

# ---- 核心静态库：模型/分词器/权重加载 ----
add_library(gpt_sft_core STATIC
  core/bpe_tokenizer.cpp
  core/safetensors.cpp
  core/gpt_sft_core.cpp
)
target_include_directories(gpt_sft_core PUBLIC core)
target_compile_options(gpt_sft_core PUBLIC ${TORCH_CXX_FLAGS})
target_link_libraries(gpt_sft_core PUBLIC ${TORCH_LIBRARIES})

# ---- 训练与推理 ----
add_subdirectory(train)
add_subdirectory(serve)
