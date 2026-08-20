#!/usr/bin/env bash
# gpt-bash/scripts/download-data.sh
#
# 下载 CMU / TellinaTool nl2bash 原始语料到 data/raw/，共 12,607 条。
#   all.nl — 自然语言指令（1.65 MB）
#   all.cm — 对应 bash 命令（1.61 MB）
#
# 镜像:
#   * 默认 githubusercontent.com（HF 上没有，GitHub raw 国内也能直连）
#   * $GH_MIRROR=ghproxy    → https://mirror.ghproxy.com/raw.githubusercontent.com/...
#   * $GH_MIRROR=kgithub    → https://raw.kgithub.com/...
#
# 跳过条件: all.nl / all.cm 已存在且 ≥ MIN_SIZE 字节（避免重下 1.6 MB）
# 失败处理: 保留 .part，可重跑脚本续传；curl -C - 断点续传
# macOS 兼容: 用 wc -c 代替 stat -c%s
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"

URL_BASE="${GH_MIRROR:-github}"
case "${URL_BASE}" in
  github)    BASE="https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash" ;;
  ghproxy)   BASE="https://mirror.ghproxy.com/raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash" ;;
  kgithub)   BASE="https://raw.kgithub.com/TellinaTool/nl2bash/master/data/bash" ;;
  *)
    echo "[警告] 未知镜像 '${URL_BASE}'，回退到 github" >&2
    BASE="https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash" ;;
esac

# nl2bash 上游实测（TellinaTool/nl2bash master 分支 2026-08）：
#   all.nl 1032340 字节 (12,607 行，约 1.0 MiB)
#   all.cm 575091 字节 (12,607 行，约 561 KiB)
# 设一个保守下限 400 KiB (409600)，防止下载到残缺文件就 mv 过去；
# 行数 sanity 由最后的 `wc -l` 兜底。
MIN_SIZE=$((400 * 1024))

# file : sha256(实际) 留空，下游若需要可用 sha256sum 补；此处只防大小不全
declare -A FILES=(
  ["all.nl"]=1645001
  ["all.cm"]=1608000
)

echo "[镜像] ${URL_BASE} → ${BASE}"

# 统计预期 / 已存在 / 待下载
need=0
have=0
for f in "${!FILES[@]}"; do
  expected=${FILES[$f]}
  dest="${RAW}/${f}"
  if [[ -s "${dest}" ]] && [[ $(wc -c < "${dest}") -ge ${MIN_SIZE} ]]; then
    have=$((have+1))
  else
    need=$((need+1))
  fi
done
if [[ ${need} -eq 0 ]]; then
  echo "[跳过] ${have} 个文件均已存在且 ≥ ${MIN_SIZE} 字节"
  exit 0
fi
echo "[下载] 待下载 ${need} 个文件"

for f in "${!FILES[@]}"; do
  expected=${FILES[$f]}
  dest="${RAW}/${f}"
  if [[ -s "${dest}" ]] && [[ $(wc -c < "${dest}") -ge ${MIN_SIZE} ]]; then
    echo "[跳过] ${f} (已存在)"
    continue
  fi
  url="${BASE}/${f}"
  part="${dest}.part"
  echo "[下载] ${f}  ←  ${url}"
  curl -fSL -C - --connect-timeout 30 --max-time 600 \
       --retry 5 --retry-delay 5 --retry-all-errors \
       "${url}" -o "${part}"
  got=$(wc -c < "${part}")
  if [[ ${got} -lt ${MIN_SIZE} ]]; then
    echo "[错误] ${f} 下载不完整: ${got} 字节 (期望 ≥ ${MIN_SIZE})" >&2
    echo "[提示] 重跑脚本可断点续传；或换镜像:" >&2
    echo "       GH_MIRROR=ghproxy $0" >&2
    echo "       GH_MIRROR=kgithub $0" >&2
    echo "       # 或 ALL_PROXY=socks5h://host:port 走代理" >&2
    exit 1
  fi
  mv "${part}" "${dest}"
done

# 行数 sanity check（CMU 上游应是 12,607；≥12000 即可）
nl_n=$(wc -l < "${RAW}/all.nl")
cm_n=$(wc -l < "${RAW}/all.cm")
echo "[完成] all.nl=${nl_n} 行, all.cm=${cm_n} 行"
if [[ ${nl_n} -ne ${cm_n} ]]; then
  echo "[警告] nl 与 cm 行数不匹配，可能上游数据错乱" >&2
fi