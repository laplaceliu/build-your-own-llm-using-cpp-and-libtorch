// reasoning/core/reasoning_core.cpp
// 推理项目核心库的占位实现：目前所有 API 都是内联 header-only。
// 保留 .cpp 以便 CMake 静态库有产物（统一打包）。
namespace reasoning {
// 预留：本文件后续会引入纯解析/序列化等与 gpt-sft 不同的工具
}

// 静态引用 gpt_sft 命名空间下的关键符号，避免静态库丢弃
#include "gpt_sft_core.h"
namespace gpt_sft_dummy {
  // 仅确保 gpt_sft 的实现被考虑到
  void keep_ref() {
    gpt_sft::config_for_size("small");
  }
}
