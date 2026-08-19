#!/usr/bin/env bash
# gpt-sft/scripts/build.sh
# 编译独立项目（训练 + 推理服务）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CXX="${CXX_COMPILER:-/usr/bin/g++-11}"
CUDA_DIR="/usr/local/cuda-13.0"

cd "${ROOT}"
cmake -B build \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_DIR}/bin/nvcc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTORCH_ROOT="${TORCH_ROOT:-/opt/libtorch}"
cmake --build build -j"$(nproc)"
echo "=== 编译完成: gpt-sft/build/{train,serve} ==="
