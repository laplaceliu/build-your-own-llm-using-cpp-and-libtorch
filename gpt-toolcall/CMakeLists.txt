# gpt-toolcall/CMakeLists.txt
#
# 模块：Function Calling / Tool Use SFT（仿 Hermes XML 风格）
# 复用 gpt-sft/core 源码，与 gpt-bash 完全平行：
#   * 共享 BPE tokenizer + safetensors loader + GPTModel
#   * 推理二进制 tool_chat 是单轮 worker（prompt in → text out）
#   * 多轮 dispatcher（tool_call 解析 + 真执行 + 回喂）由 Python 处理
#
# 这样可以不动 gpt-sft / gpt-bash 任何 C++ 代码。
cmake_minimum_required(VERSION 3.18)
project(gpt_toolcall CXX)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
  message(FATAL_ERROR "需要支持 C++17 的编译器（建议 GCC 11）")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- LibTorch（与 gpt-sft / gpt-bash 相同定位规则）----
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

# ---- nlohmann/json（gpt_sft_core 需要）----
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---- 共享的 SFT 核心 ----
# 与 gpt-bash 平行：把 gpt-sft/core 中的同名源文件编成 tool_sft_core 静态库
set(GPT_SFT_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../gpt-sft/core")
add_library(tool_sft_core STATIC
  ${GPT_SFT_CORE_DIR}/bpe_tokenizer.cpp
  ${GPT_SFT_CORE_DIR}/safetensors.cpp
  ${GPT_SFT_CORE_DIR}/gpt_sft_core.cpp
)
target_include_directories(tool_sft_core PUBLIC
  ${GPT_SFT_CORE_DIR}
)
target_compile_options(tool_sft_core PUBLIC ${TORCH_CXX_FLAGS})
target_link_libraries(tool_sft_core PUBLIC ${TORCH_LIBRARIES})
if(nlohmann_json_FOUND AND TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(tool_sft_core PUBLIC nlohmann_json::nlohmann_json)
endif()

# ---- 默认路径 ----
set(GPT_TOOLCALL_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data"
    CACHE PATH "默认数据/模型目录")
set(GPT_TOOLCALL_TOKENIZER_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../chapters/chapter02_text_data/data"
    CACHE PATH "BPE tokenizer 目录")

# ---- chat CLI（单轮 worker）----
add_executable(tool_chat chat/src/tool_chat.cpp)
target_link_libraries(tool_chat PRIVATE tool_sft_core)
target_compile_definitions(tool_chat PRIVATE
  DATA_DIR="${GPT_TOOLCALL_DATA_DIR}"
  TOKENIZER_DIR="${GPT_TOOLCALL_TOKENIZER_DIR}")

# ---- 训练说明 ----
# 训练直接复用 gpt-sft/train/sft_train 二进制（同 gpt-bash 套路）：
#   cd gpt-sft && ./scripts/build.sh build_train
#   ./gpt-sft/scripts/run.sh train \
#       --data_path ../gpt-toolcall/data/toolcall-data.json \
#       --out_dir ../gpt-toolcall/data --out_name toolcall-sft-small.pth \
#       --size small --epochs 3
#
# 数据格式: Alpaca JSON {instruction, input, output}，其中：
#   instruction = 渲染后的 prompt（system + tools + user + 之前所有 turn）
#   output      = 模型应生成的内容（tool_call XML 块 或 自然语言）
# chat_template.py 是训练/推理唯一模板源（必须 1 字节都不差）。

message(STATUS "=== gpt-toolcall build target: tool_chat ===")
message(STATUS "    完整构建: cmake --build build -j")
message(STATUS "    仅 chat:  cmake --build build --target tool_chat")