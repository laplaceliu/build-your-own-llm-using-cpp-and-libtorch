#!/usr/bin/env python3
# reasoning/scripts/convert_data_v2.py
#
# 适配 Sky-T1-17k.json (已经是 JSON 数组) + GSM8K parquet
#  - Sky-T1-17k -> sft-train.json
#  - GSM8K train -> rl-train.json
#  - GSM8K test  -> eval-test.json
#
# 输出 JSON 字段：
#   训练: {"instruction": "...", "input": "...", "output": "think...<answer>...</answer>"}
#   RL:   {"instruction": ..., "input": "", "output": full_answer_text, "gold": "123"}
#   评测: 同 RL

import argparse
import json
import os
import re

import pandas as pd


def normalize_sky_output(text: str) -> str:
    """Sky-T1 特殊标签 -> 思考/答案风格。"""
    s = text
    s = s.replace("<|begin_of_thought|>", "think")
    s = s.replace("<|end_of_thought|>", "think")
    s = s.replace("<|start_of_final_answer|>", "<answer>")
    s = s.replace("<|end_of_final_answer|>", "</answer>")
    s = s.replace("<|begin_of_solution|>", "<answer>")
    s = s.replace("<|end_of_solution|>", "</answer>")
    if "think" not in s and "<answer>" not in s:
        parts = re.split(r"\n\s*(?:The answer is|####|Answer:)\s*", s, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            s = f"think{parts[0]}think\n<answer>{parts[1].strip()}</answer>"
    return s.strip()


def sky_json_to_sft(sky_json: str, out_path: str, max_samples: int = 0) -> None:
    """Sky-T1_data_17k.json (数组, 每条 {system, conversations:[{from,value},...]}) -> SFT JSON"""
    with open(sky_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Sky-T1-17k: {len(data)} 条")
    out = []
    for row in data:
        convs = row.get("conversations", [])
        if not convs:
            continue
        user = next((m["value"] for m in convs if m.get("from") == "user"), None)
        asst = next((m["value"] for m in convs if m.get("from") == "assistant"), None)
        if not user or not asst:
            continue
        instr = str(user).strip()
        out_text = normalize_sky_output(str(asst).strip())
        if not instr or not out_text:
            continue
        out.append({"instruction": instr, "input": "", "output": out_text})
        if 0 < max_samples <= len(out):
            break
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  -> {out_path} ({len(out)} 条)")


def gsm8k_row_to_record(row) -> dict:
    q = str(row["question"]).strip()
    a = str(row["answer"]).strip()
    m = re.search(r"####\s*([-+]?[\d,\.]+)", a)
    gold = m.group(1).replace(",", "") if m else ""
    return {"instruction": q, "input": "", "output": a, "gold": gold}


def gsm8k_parquet_to_json(parquet: str, out_path: str, max_samples: int = 0) -> None:
    df = pd.read_parquet(parquet)
    out = [gsm8k_row_to_record(row) for _, row in df.iterrows()]
    if 0 < max_samples:
        out = out[:max_samples]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  -> {out_path} ({len(out)} 条)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sky-json", default="")
    p.add_argument("--gsm8k-train-parquet", default="")
    p.add_argument("--gsm8k-test-parquet", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sft-max", type=int, default=0)
    p.add_argument("--rl-max", type=int, default=500)
    p.add_argument("--eval-max", type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.sky_json and os.path.exists(args.sky_json):
        print("=== Sky-T1-17k -> SFT 训练数据 ===")
        sky_json_to_sft(args.sky_json, os.path.join(args.out_dir, "sft-train.json"), args.sft_max)
    else:
        print(f"[跳过] Sky-T1 JSON 不存在: {args.sky_json}")

    if args.gsm8k_train_parquet and os.path.exists(args.gsm8k_train_parquet):
        print("=== GSM8K train -> RL 训练数据 ===")
        gsm8k_parquet_to_json(args.gsm8k_train_parquet,
                              os.path.join(args.out_dir, "rl-train.json"),
                              args.rl_max)
    else:
        print(f"[跳过] GSM8K train parquet 不存在: {args.gsm8k_train_parquet}")

    if args.gsm8k_test_parquet and os.path.exists(args.gsm8k_test_parquet):
        print("=== GSM8K test -> 评测数据 ===")
        gsm8k_parquet_to_json(args.gsm8k_test_parquet,
                              os.path.join(args.out_dir, "eval-test.json"),
                              args.eval_max)
    else:
        print(f"[跳过] GSM8K test parquet 不存在: {args.gsm8k_test_parquet}")


if __name__ == "__main__":
    main()