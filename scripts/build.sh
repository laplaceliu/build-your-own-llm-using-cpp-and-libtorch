#!/usr/bin/env bash
# scripts/build.sh
# 配置并编译全部章节（C++ + LibTorch）
#
# 用法: ./scripts/build.sh [--clean]
#   --clean   先删除 build 目录重新配置
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cuda_env

if [ "${1:-}" = "--clean" ]; then
  echo "[清理] 删除 build/ ..."
  rm -rf "${PROJECT_ROOT}/build"
fi

echo "[配置] cmake -B build ..."
cmake -B "${PROJECT_ROOT}/build" \
  "${CMAKE_OPTIONS[@]}" \
  -DTORCH_ROOT="${LIBTORCH_ROOT}"

echo "[编译] cmake --build build -j$(nproc)"
cmake --build "${PROJECT_ROOT}/build" -j"$(nproc)"

echo ""
echo "=== 编译完成。运行某章: ./scripts/run.sh <章节名> ==="
