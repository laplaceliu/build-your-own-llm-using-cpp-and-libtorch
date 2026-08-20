# reasoning/CMakeLists.txt
# 推理模型独立项目：GPT-2 + CoT/SFT/GRPO
#
# 训练：reasoning/train/{sft_train, rl_train}      （输出 .pth）
# 推理：reasoning/serve/{reasoning_chat, reasoning_serve}
# 评测：reasoning/eval/eval_math
cmake_minimum_required(VERSION 3.16)
project(reasoning LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
  message(FATAL_ERROR "需要支持 C++17 的编译器（建议 GCC 11）")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- LibTorch 定位（与 gpt-sft 保持一致：TORCH_ROOT 环境变量 / /opt/libtorch） ----
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

# ---- 路径常量 ----
# 仓库根：reproduce libcst relative paths inside gpt-sft/cmake references
set(REASONING_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(GPT_SFT_CORE_DIR "${REPO_ROOT}/gpt-sft/core")
set(CHAPTERS_DIR "${REPO_ROOT}/chapters")
set(TOKENIZER_DIR "${CHAPTERS_DIR}/chapter02_text_data/data")

# ---- 核心静态库：复用 gpt-sft/core + 新增 reasoning/core 的实现 ----
# 复用 gpt-sft 的源码（gpt-sft 不被独立构建，由 reasoning 接管）
add_library(reasoning_core STATIC
  ${GPT_SFT_CORE_DIR}/bpe_tokenizer.cpp
  ${GPT_SFT_CORE_DIR}/safetensors.cpp
  ${GPT_SFT_CORE_DIR}/gpt_sft_core.cpp
  core/reasoning_core.cpp
  core/sft_train.cpp
  core/rl_train.cpp
)
target_include_directories(reasoning_core PUBLIC
  core
  ${GPT_SFT_CORE_DIR}
  ${CHAPTERS_DIR}/chapter04_gpt/include
  ${CHAPTERS_DIR}/chapter05_pretraining/include
  ${CHAPTERS_DIR}/chapter07_instruction_tuning/include
)
target_compile_options(reasoning_core PUBLIC ${TORCH_CXX_FLAGS})
target_link_libraries(reasoning_core PUBLIC ${TORCH_LIBRARIES})

# ---- 子模块 ----
add_subdirectory(train)
add_subdirectory(serve)
add_subdirectory(eval)
