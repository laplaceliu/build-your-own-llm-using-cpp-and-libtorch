#!/usr/bin/env bash
# reasoning/scripts/download-data.sh
# 下载推理训练/评测数据：
#   - Sky-T1-17k (NovaSky-AI/Sky-T1_data-17k) -> SFT 训练数据（转 思考/答案 格式）
#   - GSM8K (openai/gsm8k) -> RL 训练子集 + 评测子集
#
# 走 hf-mirror.com 镜像（大陆可用）；失败回退 huggingface.co + 代理
# 输出：reasoning/data/{sft-train.json, rl-train.json, rl-test.json}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/data"
mkdir -p "${DATA_DIR}/downloads"
cd "${DATA_DIR}/downloads"

# 代理设置（仅 huggingface.co 失败时启用）
PROXY_ENV=()
if [ -n "${HF_HTTPS_PROXY:-${HTTPS_PROXY:-}}" ]; then
  PROXY_ENV=("HTTPS_PROXY=${HF_HTTPS_PROXY:-${HTTPS_PROXY:-}}")
fi

# 镜像备用列表
MIRRORS=("https://hf-mirror.com" "https://huggingface.co")

download_with_mirror() {
  local url_path="$1"  # 形如 datasets/NovaSky-AI/Sky-T1_data_17k/resolve/main/data/train-00000-of-00001-???...parquet
  local outfile="$2"
  for mirror in "${MIRRORS[@]}"; do
    local url="${mirror}/${url_path}"
    echo "  尝试 ${url}"
    if env "${PROXY_ENV[@]}" curl -fL --connect-timeout 20 --retry 2 -o "${outfile}" "${url}" 2>/dev/null; then
      echo "  [成功] ${url}"
      return 0
    fi
    echo "  [失败] ${mirror}"
  done
  return 1
}

# 检查 python 依赖
python3 -c "import pandas, pyarrow" 2>/dev/null || {
  echo "缺少 Python 依赖，请运行: pip install pandas pyarrow"
  exit 1
}

# ============================================================================
# 1) Sky-T1-17k（推理 SFT 数据，约 17k 样本）
#    原仓库：NovaSky-AI/Sky-T1_data_17k  （有 parquet 文件）
# ============================================================================
echo "=== 下载 Sky-T1-17k ==="
SKY_PARQUET="sky-t1-17k.parquet"
if [ ! -f "${SKY_PARQUET}" ]; then
  # 尝试常见的 parquet 路径
  if ! download_with_mirror "datasets/NovaSky-AI/Sky-T1_data_17k/resolve/main/data/train-00000-of-00001.parquet" "${SKY_PARQUET}"; then
    echo "Sky-T1-17k 直连失败，尝试下载 JSON 版本..."
    if ! download_with_mirror "datasets/NovaSky-AI/Sky-T1_data_17k/resolve/main/sky-t1-17k.json" "sky-t1-17k.json"; then
      echo "错误: 无法下载 Sky-T1-17k，请手动放置到 ${DATA_DIR}/downloads/${SKY_PARQUET}"
      exit 1
    fi
  fi
fi

# ============================================================================
# 2) GSM8K（数学题 7.5k train / 1.3k test）
# ============================================================================
echo "=== 下载 GSM8K ==="
GSM8K_TRAIN_PARQUET="gsm8k-train.parquet"
GSM8K_TEST_PARQUET="gsm8k-test.parquet"
if [ ! -f "${GSM8K_TRAIN_PARQUET}" ]; then
  download_with_mirror "datasets/openai/gsm8k/resolve/main/gsm8k/main/train-00000-of-00001.parquet" "${GSM8K_TRAIN_PARQUET}" || {
    echo "错误: 无法下载 GSM8K train"
    exit 1
  }
fi
if [ ! -f "${GSM8K_TEST_PARQUET}" ]; then
  download_with_mirror "datasets/openai/gsm8k/resolve/main/gsm8k/main/test-00000-of-00001.parquet" "${GSM8K_TEST_PARQUET}" || {
    echo "错误: 无法下载 GSM8K test"
    exit 1
  }
fi

# ============================================================================
# 3) 转换：parquet -> reasoning JSON 训练格式
# ============================================================================
echo "=== 转换数据格式 ==="
python3 "${ROOT}/scripts/convert_data.py" \
  --sky-parquet "${SKY_PARQUET}" \
  --gsm8k-train-parquet "${GSM8K_TRAIN_PARQUET}" \
  --gsm8k-test-parquet "${GSM8K_TEST_PARQUET}" \
  --out-dir "${DATA_DIR}"

echo "=== 下载完成 ==="
ls -la "${DATA_DIR}"/*.json
