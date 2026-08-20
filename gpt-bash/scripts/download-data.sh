#!/usr/bin/env bash
# gpt-bash/scripts/download-data.sh
#
# 下载 nl2bash-shell（功能等价于 TellinaTool/nl2bash 公开版本）。
# 数据描述见 data/README.md。
#
# 行为:
#   1) 若已存在 gpt-bash/data/raw/all.{nl,cm} 则跳过
#   2) 否则从 https://github.com/TellinaTool/nl2bash/data/bash 下载 all.nl 和 all.cm
#
# 网络: 必要时设置 ALL_PROXY / HTTPS_PROXY。
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"

URL_BASE="https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash"

cd "${RAW}"
for f in all.nl all.cm; do
  if [[ -s "${f}" ]]; then
    echo "[跳过] ${f} 已存在 (大小 $(stat -c%s "${f}"))"
  else
    echo "[下载] ${URL_BASE}/${f}"
    curl -fSL --max-time 60 "${URL_BASE}/${f}" -o "${f}"
    echo "[完成] ${f} (大小 $(stat -c%s "${f}"))"
  fi
done

echo
echo "=== 数据概要 ==="
wc -l all.nl all.cm
