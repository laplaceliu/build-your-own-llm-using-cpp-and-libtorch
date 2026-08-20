# gpt-bash/CMakeLists.txt
#
# 模块：自然语言 → Bash 命令生成 SFT
# 直接复用 gpt-sft/core 的实现源文件（bpe_tokenizer.cpp, safetensors.cpp,
# gpt_sft_core.cpp），编译成一个静态库 bash_sft_core，与 gpt-sft 的
# gpt_sft_core 二进制兼容（同源同名函数 + 命名空间）。
#
# 这样 gpt-bash 既可以独立构建，也与 gpt-sft 保持同步。
cmake_minimum_required(VERSION 3.18)
project(gpt_bash CXX)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
  message(FATAL_ERROR "需要支持 C++17 的编译器（建议 GCC 11）")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- LibTorch (与 gpt-sft 相同的定位规则) ----
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

# ---- nlohmann/json（gpt_sft_core 需要，用于解析 instruction-data.json） ----
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  # 项目内通常已经安装；找系统头文件 fallback
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---- 共享的 SFT 核心 ----
# 复用 gpt-sft/core 中的同名源文件，编译成 bash_sft_core 静态库。
# 这样 gpt-bash 完全自包含，不依赖 gpt-sft 的构建状态。
set(GPT_SFT_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../gpt-sft/core")
add_library(bash_sft_core STATIC
  ${GPT_SFT_CORE_DIR}/bpe_tokenizer.cpp
  ${GPT_SFT_CORE_DIR}/safetensors.cpp
  ${GPT_SFT_CORE_DIR}/gpt_sft_core.cpp
)
target_include_directories(bash_sft_core PUBLIC
  ${GPT_SFT_CORE_DIR}   # 让 include "gpt_sft_core.h" 找得到
)
target_compile_options(bash_sft_core PUBLIC ${TORCH_CXX_FLAGS})
target_link_libraries(bash_sft_core PUBLIC ${TORCH_LIBRARIES})
if(nlohmann_json_FOUND AND TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(bash_sft_core PUBLIC nlohmann_json::nlohmann_json)
endif()

# ---- 默认路径 ----
set(GPT_BASH_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data"
    CACHE PATH "默认数据/模型目录")
set(GPT_BASH_TOKENIZER_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../chapters/chapter02_text_data/data"
    CACHE PATH "BPE tokenizer 目录")

# ---- chat CLI ----
add_executable(bash_chat chat/src/bash_chat.cpp)
target_link_libraries(bash_chat PRIVATE bash_sft_core)
target_compile_definitions(bash_chat PRIVATE
  DATA_DIR="${GPT_BASH_DATA_DIR}"
  TOKENIZER_DIR="${GPT_BASH_TOKENIZER_DIR}")

# ---- 训练说明 ----
# 训练直接复用 gpt-sft/train/sft_train 二进制：
#   cd gpt-sft && ./scripts/build.sh build_train
#   ./gpt-sft/scripts/run.sh train \
#       --data_path ../gpt-bash/data/bash-instruction-data.json \
#       --out_dir ../gpt-bash/data --out_name bash-sft-small.pth \
#       --size small --epochs 5
# 不需要把 bash_chat 显式依赖 sft_train，避免构建顺序耦合。

message(STATUS "=== gpt-bash build target: bash_chat ===")
message(STATUS "    完整构建: cmake --build build -j")
message(STATUS "    仅 chat:  cmake --build build --target bash_chat")
