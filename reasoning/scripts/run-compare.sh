#!/usr/bin/env bash
# reasoning/scripts/run-compare.sh
# 一键编排：SFT(small/medium) → RL(small/medium) → 评测 → 报告
#
# 输入：reasoning/data/{sft-train.json, rl-train.json, eval-test.json}
# 输出：reasoning/data/{small,medium}-{sft,rl,sft_rl}.pth
#       reasoning/data/compare.csv  +  reasoning/docs/compare-report.md
#
# 训练时长观察：本脚本会记录每个训练阶段开始/结束时间。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
DOCS="${ROOT}/docs"
TIME_LOG="${DATA}/compare-times.log"
mkdir -p "${DATA}" "${DOCS}"

# 预设权重路径（缺失则跳过对应 RL 训练）
SMALL_BASE="${DATA}/gpt2-small.safetensors"
MED_BASE="${DATA}/gpt2-medium.safetensors"

# 检测 GPU 用于日志
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo CPU)"
GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
echo "GPU: ${GPU_NAME} (${GPU_MEM})" | tee -a "${TIME_LOG}"

stamp() { date '+%F %T'; }
step_log() {
  echo "[$(stamp)] $*" | tee -a "${TIME_LOG}"
}

step_log "=== 对比实验开始 ==="

# 1. SFT 小模型
if [ -f "${DATA}/sft-train.json" ]; then
  step_log "SFT small 开始"
  T0=$(date +%s); ./scripts/run.sh sft_train --data "${DATA}/sft-train.json" \
    --size small --weights "${SMALL_BASE}" \
    --epochs 1 --batch 4 --max-length 1024 \
    --out "${DATA}/small-sft.pth" 2>&1 | tee "${DATA}/sft-small.log" | tail -5
  step_log "SFT small 完成，耗时 $(($(date +%s) - T0))s"
fi

# 2. SFT 中模型
if [ -f "${DATA}/sft-train.json" ]; then
  step_log "SFT medium 开始"
  T0=$(date +%s); ./scripts/run.sh sft_train --data "${DATA}/sft-train.json" \
    --size medium --weights "${MED_BASE}" \
    --epochs 1 --batch 2 --max-length 1024 \
    --out "${DATA}/medium-sft.pth" 2>&1 | tee "${DATA}/sft-medium.log" | tail -5
  step_log "SFT medium 完成，耗时 $(($(date +%s) - T0))s"
fi

# 3. RL 小模型（从基座）
if [ -f "${DATA}/rl-train.json" ]; then
  step_log "RL small (from base) 开始"
  T0=$(date +%s); ./scripts/run.sh rl_train --data "${DATA}/rl-train.json" \
    --size small --weights "${SMALL_BASE}" \
    --max-steps 30 --batch 2 --group 4 --max-new 256 \
    --lr 1e-6 --out "${DATA}/small-rl.pth" 2>&1 | tee "${DATA}/rl-small.log" | tail -5
  step_log "RL small 完成，耗时 $(($(date +%s) - T0))s"
fi

# 4. RL 中模型（从基座）
if [ -f "${DATA}/rl-train.json" ]; then
  step_log "RL medium (from base) 开始"
  T0=$(date +%s); ./scripts/run.sh rl_train --data "${DATA}/rl-train.json" \
    --size medium --weights "${MED_BASE}" \
    --max-steps 20 --batch 1 --group 3 --max-new 256 \
    --lr 1e-6 --out "${DATA}/medium-rl.pth" 2>&1 | tee "${DATA}/rl-medium.log" | tail -5
  step_log "RL medium 完成，耗时 $(($(date +%s) - T0))s"
fi

# 5. SFT+RL：从 SFT 继续 RL
if [ -f "${DATA}/rl-train.json" ] && [ -f "${DATA}/small-sft.pth" ]; then
  step_log "RL small (from sft) 开始"
  T0=$(date +%s); ./scripts/run.sh rl_train --data "${DATA}/rl-train.json" \
    --size small --init "${DATA}/small-sft.pth" \
    --max-steps 20 --batch 2 --group 4 --max-new 256 \
    --lr 5e-7 --out "${DATA}/small-sft_rl.pth" 2>&1 | tee "${DATA}/rl-sft-small.log" | tail -5
  step_log "RL small (from sft) 完成，耗时 $(($(date +%s) - T0))s"
fi

# 6. 评测对比
if [ -f "${DATA}/eval-test.json" ]; then
  step_log "评测对比开始"
  MODELS=""
  NAMES=""
  for x in "${SMALL_BASE}" "${DATA}/small-sft.pth" "${DATA}/small-rl.pth" "${DATA}/small-sft_rl.pth" \
           "${MED_BASE}" "${DATA}/medium-sft.pth" "${DATA}/medium-rl.pth"; do
    if [ -f "$x" ]; then
      MODELS="${MODELS:+,}${x}"
      nm="$(basename "$x" .pth)"; nm="$(basename "$nm" .safetensors)"
      NAMES="${NAMES:+,}${nm}"
    fi
  done
  ./scripts/run.sh eval_math --data "${DATA}/eval-test.json" \
    --size small \
    --models "${MODELS}" --names "${NAMES}" \
    --strategies greedy,vote \
    --samples 3 --max-new 256 \
    --out "${DATA}/compare.csv" 2>&1 | tee "${DATA}/eval-small.log" | tail -10
  ./scripts/run.sh eval_math --data "${DATA}/eval-test.json" \
    --size medium \
    --models "${MODELS}" --names "${NAMES}" \
    --strategies greedy,vote \
    --samples 3 --max-new 256 \
    --out "${DATA}/compare-medium.csv" 2>&1 | tee "${DATA}/eval-medium.log" | tail -10
  step_log "评测对比完成"
fi

# 7. 汇总报告
step_log "生成对比报告"
python3 "${ROOT}/scripts/report.py" \
  --csv "${DATA}/compare.csv" \
  --csv "${DATA}/compare-medium.csv" \
  --out "${DOCS}/compare-report.md" \
  --time-log "${TIME_LOG}" || echo "  报告生成失败"

step_log "=== 对比实验完成 ==="
echo "结果: ${DATA}/compare.csv, ${DOCS}/compare-report.md"
