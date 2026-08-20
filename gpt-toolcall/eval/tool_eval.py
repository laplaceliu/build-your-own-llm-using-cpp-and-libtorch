#!/usr/bin/env python3
"""tool_eval.py — Function Calling 评测（不调真工具，只看模型输出是否合法）

评测任务（每条 Alpaca 样本）：
  * instruction = 渲染后的 prompt（含 USER + 历史）
  * expected_output = 真值（assistant 应写的内容）
  * 评分维度（每个 0/1）：
    1. **valid_tool_call**: 期望有 <tool_call> 时，模型输出含合法 JSON + 已知工具名
    2. **correct_tool**: 调用的工具名与真值一致（仅在 valid 时检查）
    3. **correct_args**: arguments JSON keys 与真值一致（不检查值，模型可能填错）
    4. **natural_language**: 期望自然语言时，输出不含 <tool_call>

用法:
  python3 eval/tool_eval.py \
    --data data/smoke/toolcall-smoke.json \
    --binary build/tool_chat \
    --model data/toolcall-sft-medium.pth \
    --size medium \
    --out logs/eval.json \
    [--limit 12]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR.parent / "chat"))
import chat_template as ct  # noqa: E402
from dispatch import extract_tool_call  # noqa: E402

TOOL_NAMES = {"get_weather", "get_time", "calc"}  # 与 dispatch.py 注册的工具一致


def call_worker(binary, prompt, model, size, max_new=128, no_cuda=False) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(prompt)
        p = f.name
    try:
        cmd = [binary, "--model", model, "--size", size,
               "--prompt-file", p, "--max-new", str(max_new), "--json"]
        if no_cuda:
            cmd.append("--no-cuda")
        env = os.environ.copy()
        tr = os.environ.get("TORCH_ROOT", "/opt/libtorch")
        env["LD_LIBRARY_PATH"] = f"{tr}/lib:" + env.get("LD_LIBRARY_PATH", "")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        if proc.returncode != 0:
            return {"text": f"[ERROR] {proc.stderr[:200]}", "stopped_on": "error"}
        return json.loads(proc.stdout)
    finally:
        try: os.unlink(p)
        except FileNotFoundError: pass


def score(expected: str, actual: str) -> dict:
    """单条评分。返回 {valid_tc, correct_tool, correct_args, natural_language, is_tc_expected}。"""
    expected_tc = extract_tool_call(expected)
    actual_tc = extract_tool_call(actual)
    res = {
        "valid_tc": 0, "correct_tool": 0, "correct_args": 0,
        "natural_language": 0,
        "is_tc_expected": 1 if expected_tc else 0,
    }
    if expected_tc:
        # 期望 tool_call
        if actual_tc:
            res["valid_tc"] = 1
            if actual_tc["name"] == expected_tc["name"]:
                res["correct_tool"] = 1
            exp_args = expected_tc.get("arguments", {})
            act_args = actual_tc.get("arguments", {})
            if set(exp_args.keys()) == set(act_args.keys()):
                res["correct_args"] = 1
    else:
        # 期望自然语言
        if not actual_tc:
            res["natural_language"] = 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Alpaca JSON 文件（每行 {instruction, output}）")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", default="medium")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="写出 JSON 评测结果")
    args = ap.parse_args()

    data = []
    with open(args.data) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                data.append(json.loads(ln))
    if args.limit > 0:
        data = data[:args.limit]

    n = len(data)
    s = {"valid_tc": 0, "correct_tool": 0, "correct_args": 0, "natural_language": 0}
    n_tc_exp = 0
    n_nl_exp = 0
    results = []
    for i, sample in enumerate(data):
        instr = sample["instruction"]
        expected = sample["output"]
        out = call_worker(args.binary, instr, args.model, args.size,
                          max_new=args.max_new, no_cuda=args.no_cuda)
        actual = out["text"]
        sc = score(expected, actual)
        for k in s:
            s[k] += sc[k]
        if sc["is_tc_expected"]:
            n_tc_exp += 1
        else:
            n_nl_exp += 1
        results.append({
            "idx": i,
            "instruction_tail": instr[-80:],
            "expected": expected[:120],
            "actual": actual[:120],
            "scores": sc,
        })
        # 进度
        print(f"[{i+1}/{n}] tc={sc['is_tc_expected']} valid={sc['valid_tc']} "
              f"tool={sc['correct_tool']} args={sc['correct_args']} "
              f"nl={sc['natural_language']}  actual={actual[:60]!r}")

    print("\n--- 总体 ---")
    print(f"样本数: {n} (expect tool_call: {n_tc_exp}, expect NL: {n_nl_exp})")
    if n_tc_exp > 0:
        print(f"valid_tc:       {s['valid_tc']}/{n_tc_exp} = {s['valid_tc']/n_tc_exp:.1%}")
        print(f"correct_tool:   {s['correct_tool']}/{n_tc_exp} = {s['correct_tool']/n_tc_exp:.1%}")
        print(f"correct_args:   {s['correct_args']}/{n_tc_exp} = {s['correct_args']/n_tc_exp:.1%}")
    if n_nl_exp > 0:
        print(f"natural_lang:   {s['natural_language']}/{n_nl_exp} = {s['natural_language']/n_nl_exp:.1%}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "summary": {**s, "n": n, "n_tc_exp": n_tc_exp, "n_nl_exp": n_nl_exp},
                "samples": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[写出] {args.out}")


if __name__ == "__main__":
    main()