#!/usr/bin/env bash
# scripts/common.sh
# 公共环境与工具函数：项目根路径、运行时环境变量、下载函数。
# 用法：source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
set -euo pipefail

# ---- 项目根目录（本脚本位于 <root>/scripts/ 下）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- 计算设备与 CUDA 版本（可按机器修改）----
CUDA_VERSION="${CUDA_VERSION:-13.0}"
LIBTORCH_ROOT="${TORCH_ROOT:-/opt/libtorch}"
CUDA_DIR="/usr/local/cuda-${CUDA_VERSION}"

# ---- 运行时链接库（source 后生效）----
export LD_LIBRARY_PATH="${LIBTORCH_ROOT}/lib:${CUDA_DIR}/lib64:${CUDA_DIR}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 编译配置（build.sh 使用）
CMAKE_OPTIONS=(
  "-DCMAKE_CXX_COMPILER=${CXX_COMPILER:-/usr/bin/g++-11}"
  "-DCMAKE_CUDA_COMPILER=${CUDA_DIR}/bin/nvcc"
  "-DCMAKE_BUILD_TYPE=Release"
)

# 供 build.sh / run.sh 使用的编译期 CUDA 环境
cuda_env() {
  export PATH="${CUDA_DIR}/bin:${PATH}"
  export CUDA_ROOT="${CUDA_DIR}" CUDA_HOME="${CUDA_DIR}"
  export CUDAToolkit_ROOT="${CUDA_DIR}" CUDA_TOOLKIT_ROOT_DIR="${CUDA_DIR}"
}

# 清理可能残留的跨平台编译变量（见项目记忆：256make 残留会污染）
clean_cross_env() {
  unset LD_LIBRARY_PATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH CPATH LIBRARY_PATH CC CXX \
        CFLAGS CXXFLAGS QMAKE_CC QMAKE_CXX QMAKE_LINK 2>/dev/null || true
}

# ---- 下载函数：优先直连（hf-mirror 无代理更快），失败回退代理 ----
# download <url> <输出文件> <描述>
download() {
  local url="$1" out="$2" desc="$3"
  if [ -f "${out}" ]; then
    echo "  [跳过] ${desc} 已存在: ${out}"
    return 0
  fi
  mkdir -p "$(dirname "${out}")"
  echo "  [下载] ${desc} -> ${out}"
  if curl -fsSL --connect-timeout 20 --retry 2 -o "${out}" "${url}" 2>/dev/null; then
    echo "  [完成] 直连"
  elif curl -fsSL --connect-timeout 20 --retry 1 -x http://localhost:8888 \
       -o "${out}" "${url}" 2>/dev/null; then
    echo "  [完成] 代理回退"
  else
    echo "  [失败] ${url}"
    rm -f "${out}"
    return 1
  fi
}

# 代理直通变量（下载大文件时若 hf-mirror 慢，可尝试换源）
proxy_vars() {
  export HTTP_PROXY=http://localhost:8888 HTTPS_PROXY=http://localhost:8888 \
         ALL_PROXY=socks5h://localhost:30000
}
