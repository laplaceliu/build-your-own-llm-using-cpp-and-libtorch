#!/usr/bin/env python3
# reasoning/scripts/convert_data.py
#
# 下载后的原始数据转换为本项目训练用 JSON 格式：
#   - Sky-T1-17k  -> sft-train.json  (含 思考/答案 标签)
#   - GSM8K train -> rl-train.json   (RL 训练样本)
#   - GSM8K test  -> eval-test.json   (评测样本)
#
# 各 JSON 字段格式：
#   训练: {"instruction": "...", "input": "...", "output": "think...<answer>...</answer>"}
#   评测: {"instruction": "...", "input": "...", "output": "#### ANSWER"}   (GSM8K 原始格式)
import argparse
import json
import os
import re

import pandas as pd


def sky_to_sft(sky_parquet: str, out_path: str, max_samples: int = 0) -> None:
    """读取 Sky-T1 parquet，转为推理 SFT 训练数据。

    适配多种 Sky-T1 字段命名（origin / horizon / 17k / 等）：
      - messages 角色式  -> 抽取 user 问 / assistant 答
      - question / answer  -> 直接对应 instruction / output
      - 带有 特殊 thought 标签  -> 转换到 思考/答案 风格
    """
    df = pd.read_parquet(sky_parquet)
    print(f"  Sky-T1 列: {list(df.columns)}, 样本数: {len(df)}")

    out = []
    for _, row in df.iterrows():
        item = _sky_row_to_entry(row)
        if item is None:
            continue
        out.append(item)
        if 0 < max_samples <= len(out):
            break

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  -> {out_path} ({len(out)} 条)")


def _sky_row_to_entry(row) -> dict | None:
    """单条 Sky-T1 -> {"instruction":..., "input":..., "output":...}"""
    # 1) messages 风格（OpenAI chat）
    if "messages" in row:
        msgs = row["messages"]
        if isinstance(msgs, (list, tuple)) and len(msgs) >= 2:
            user = next((m for m in msgs if m.get("role") == "user"), None)
            asst = next((m for m in msgs if m.get("role") == "assistant"), None)
            if user and asst:
                instr = str(user.get("content", "")).strip()
                out = _normalize_sky_output(str(asst.get("content", "")).strip())
                if instr and out:
                    return {"instruction": instr, "input": "", "output": out}

    # 2) 字段命名猜测
    q = _get_field(row, ["question", "instruction", "prompt", "query"])
    a = _get_field(row, ["answer", "output", "response", "solution", "final_answer"])
    if q and a:
        return {"instruction": str(q).strip(), "input": "", "output": _normalize_sky_output(str(a))}

    return None


def _get_field(row, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in row:
            v = row[c]
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return v
    return None


def _normalize_sky_output(text: str) -> str:
    """把 Sky-T1 的特殊 thought 标签转成 思考/答案 风格。"""
    s = text
    s = s.replace("<|begin_of_thought|>", "think")
    s = s.replace("<|end_of_thought|>", "think")
    s = s.replace("<|start_of_final_answer|>", "<answer>")
    s = s.replace("<|end_of_final_answer|>", "</answer>")
    s = s.replace("<|begin_of_solution|>", "<answer>")
    s = s.replace("<|end_of_solution|>", "</answer>")
    # 如果没有 answer 标签，简单包一层
    if "think" not in s and "<answer>" not in s:
        # 有可能是一段连续推理+答案，用 首个换行/句号 切分
        parts = re.split(r"\n\s*(?:The answer is|####|Answer:)\s*", s, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            s = f"think{parts[0]}think\n<answer>{parts[1].strip()}</answer>"
    return s.strip()


def gsm8k_to_record(row, gold_field: str | None = None) -> dict:
    """GSM8K 行 -> {"instruction": question, "input": "", "output": full_answer, "gold": gold_answer}"""
    q = str(row["question"]).strip()
    a = str(row["answer"]).strip()
    # 抽取 GSM8K 答案：#### 后的数字
    m = re.search(r"####\s*([-+]?[\d,\.]+)", a)
    gold = m.group(1).replace(",", "") if m else ""
    return {"instruction": q, "input": "", "output": a, "gold": gold}


def gsm8k_to_rl(parquet: str, out_path: str, max_samples: int = 500) -> None:
    """RL 训练用：题目 + 答案完整文本（用于奖励计算）"""
    df = pd.read_parquet(parquet)
    out = [gsm8k_to_record(row) for _, row in df.head(max_samples).iterrows()]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  -> {out_path} ({len(out)} 条)")


def gsm8k_to_eval(parquet: str, out_path: str, max_samples: int = 100) -> None:
    """评测用：题目 + gold 答案"""
    df = pd.read_parquet(parquet)
    out = [gsm8k_to_record(row) for _, row in df.head(max_samples).iterrows()]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  -> {out_path} ({len(out)} 条)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sky-parquet", required=True)
    p.add_argument("--gsm8k-train-parquet", required=True)
    p.add_argument("--gsm8k-test-parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sft-max", type=int, default=0, help="SFT 样本上限，0 表示全部")
    p.add_argument("--rl-max", type=int, default=500)
    p.add_argument("--eval-max", type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Sky-T1-17k -> SFT 训练数据 ===")
    if os.path.exists(args.sky_parquet):
        sky_to_sft(args.sky_parquet, os.path.join(args.out_dir, "sft-train.json"), args.sft_max)
    else:
        print(f"  跳过: {args.sky_parquet} 不存在")

    print("=== GSM8K train -> RL 训练数据 ===")
    gsm8k_to_rl(args.gsm8k_train_parquet, os.path.join(args.out_dir, "rl-train.json"), args.rl_max)

    print("=== GSM8K test -> 评测数据 ===")
    gsm8k_to_eval(args.gsm8k_test_parquet, os.path.join(args.out_dir, "eval-test.json"), args.eval_max)


if __name__ == "__main__":
    main()
