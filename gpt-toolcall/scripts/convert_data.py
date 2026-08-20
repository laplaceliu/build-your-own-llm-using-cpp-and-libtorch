#!/usr/bin/env python3
"""convert_data.py — Hermes JSONL → Alpaca JSON

输入（Hermes-function-calling-v1）:
  每行 JSON，形如:
    {
      "conversations": [
        {"from": "system",    "value": "..."},
        {"from": "human",     "value": "..."},
        {"from": "gpt",       "value": "..."},
        {"from": "tool",      "value": "..."},
        {"from": "gpt",       "value": "..."},
        ...
      ]
    }

输出（Alpaca 格式，可直接喂给 gpt-sft/train/sft_train）:
  [
    {"instruction": "<system + tools + user + 之前所有 turn 的渲染>",
     "output":     "<gpt 的下一段内容，可能是 <tool_call>...</tool_call> 或自然语言>",
     "_meta": {"tools_used": ["..."], "turns": N}
    },
    ...
  ]

为什么这么干：
  * gpt-sft/core/instruction.h 的 InstructionEntry 只有 instruction/input/output 三字段，
    不支持多轮 / tool_call。改 C++ 训练代码是大动作（动 dataloader、format_input 等）。
  * Hermes 多轮对话本质是"看到一个 history，预测下一条 assistant 回复"。
    把 history 渲染成 instruction，把"下一条"作为 output，就退化成单条 SFT。
  * 这与 Hermes 官方训练脚本 `tool_calling.py` 的处理方式一致（last-turn-only 渲染）。

用法：
  python3 scripts/convert_data.py \
    --in data/raw/hermes-train.jsonl \
    --out data/toolcall-data.json \
    --shuffle --seed 0
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR.parent / "chat"))
import chat_template as ct  # noqa: E402

# Hermes 的 system value 中可能含工具描述（`<tools>...</tools>` 块）
# 我们既要把它们剥离出来，也要把 system 中不含工具描述的部分用作 system 文本
TOOLS_BLOCK_RE = re.compile(re.escape(ct.TOOLS_OPEN) + r"\s*(.*?)\s*" + re.escape(ct.TOOLS_CLOSE), re.DOTALL)


def normalize_tool(t: dict) -> dict:
    """把 OpenAI-style {type:function, function:{name,...}} 拍成 {name, description, parameters}。
    对扁平 {name, description, parameters} 则原样返回。
    """
    if "function" in t and isinstance(t["function"], dict):
        fn = t["function"]
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }
    return t


def parse_hermes_tools(system_value: str) -> tuple[list[dict], str]:
    """从 system 文本里抠出 <tools>...</tools> 块，返回 (tools, system_without_tools)。

    实际 Hermes-function-calling-v1 系统块格式有两种：
      1. <tools>[{"type":"function","function":{...}}, ...]</tools>
         （整个 JSON 数组，无 <tool> 嵌套）
      2. <tools>
           <tool>{...}</tool>
           <tool>{...}</tool>
         </tools>
         （Hermes XML 标准格式）
    """
    m = TOOLS_BLOCK_RE.search(system_value)
    if not m:
        return [], system_value.strip()
    inner = m.group(1).strip()
    tools: list[dict] = []
    if inner.startswith("["):
        # OpenAI JSON 数组
        try:
            arr = json.loads(inner)
            for t in arr:
                tools.append(normalize_tool(t))
        except json.JSONDecodeError:
            pass
    else:
        # Hermes XML 多 <tool>...</tool>
        for tm in re.finditer(r"<tool>(.*?)</tool>", inner, re.DOTALL):
            try:
                tools.append(normalize_tool(json.loads(tm.group(1))))
            except json.JSONDecodeError:
                continue
    rest = (system_value[:m.start()] + system_value[m.end():]).strip()
    return tools, rest


def hermes_role_to_msg(role: str, value: str) -> dict:
    """Hermes from → messages 数组的 role 字段。"""
    role_map = {
        "system":    "system",   # 通常只有第一条
        "human":     "user",
        "user":      "user",
        "gpt":       "assistant",
        "assistant": "assistant",
        "tool":      "tool",
        "observation": "tool",   # 同义
    }
    r = role_map.get(role.lower())
    if r is None:
        raise ValueError(f"未知 Hermes role: {role}")
    return {"role": r, "content": value}


def render_messages_with_tools(messages: list[dict], tools: list[dict],
                                fallback_system: str | None = None) -> str:
    """拼接完整的 prompt：system（含 tool 描述，从 Hermes system 中抠出）+ tools + messages。

    如果 Hermes 没有 system，用 fallback_system。
    """
    # 找到 / 注入 system
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    if not system:
        system = fallback_system or (
            "You are a helpful assistant with access to tools. "
            "When a tool is needed, respond with a <tool_call> block. "
            "When a natural-language answer suffices, respond directly."
        )
    # 去掉 messages 里第一个 system（render_full_prompt 会再补 system）
    msgs_no_system = [m for m in messages if m["role"] != "system"]
    return ct.render_full_prompt(msgs_no_system, tools=tools, system=system)


def expand_one_sample(sample: dict, fallback_system: str | None) -> list[dict]:
    """把一条多轮样本展开为多条 (instruction, output) 对。

    关键：每条 assistant/gpt turn 之前的所有内容（含 system + tools + 所有之前的 human/tool）
    渲染为 instruction；assistant 该 turn 的 value 作为 output。
    """
    convs = sample.get("conversations") or sample.get("messages")
    if not convs:
        return []

    # 第一遍：把 system 中的 <tools>...</tools> 抠出来
    # 同时合并 sample["tools"] 字段（Hermes-function-calling-v1 主数据集就是这种结构：
    # 系统 prompt 是 "you may call one or more functions"，<tools></tools> 占位，
    # 真正的工具列表在 sample["tools"] 这个独立的顶层字段里 —— 实测是字符串形式的 JSON 数组）
    tools: list[dict] = []
    raw_tools = sample.get("tools")
    if isinstance(raw_tools, list):
        for t in raw_tools:
            tools.append(normalize_tool(t))
    elif isinstance(raw_tools, str) and raw_tools.strip():
        try:
            arr = json.loads(raw_tools)
            for t in arr:
                tools.append(normalize_tool(t))
        except json.JSONDecodeError:
            pass

    cleaned_convs = []
    for c in convs:
        role = c.get("from") or c.get("role")
        value = c.get("value", c.get("content", ""))
        if role in ("system", "SYSTEM") and isinstance(value, str):
            sys_tools, sys_rest = parse_hermes_tools(value)
            if sys_tools:
                tools = sys_tools   # 优先用 system 里的（更新）
            value = sys_rest
        cleaned_convs.append({"role": role, "content": value})

    if not tools:
        # 没有工具描述 → 这条样本不能用作 function calling 训练，跳过
        return []

    # 第二遍：每遇到 assistant/gpt 就生成一条训练样本
    # 第一条 assistant 也是有效样本（system + tools + user → assistant 第一句话）
    # cleaned_convs 里 role 是 Hermes 原始 from 字段（system/human/gpt/tool），
    # 还没经过 hermes_role_to_msg 映射，这里直接按值判断
    out = []
    messages_so_far: list[dict] = []
    for c in cleaned_convs:
        role = c["role"].lower()
        value = c["content"]
        if role in ("gpt", "assistant"):
            instr = render_messages_with_tools(messages_so_far, tools, fallback_system)
            out.append({
                "instruction": instr,
                "input": "",
                "output": value,
                "_meta": {
                    "tools_used": [t["name"] for t in tools],
                    "turns": len(messages_so_far) + 1,
                },
            })
            messages_so_far.append({"role": "assistant", "content": value})
        elif role in ("human", "user"):
            messages_so_far.append({"role": "user", "content": value})
        elif role in ("tool", "observation"):
            messages_so_far.append({"role": "tool", "content": value})
        # system 已经在 render_messages_with_tools 里处理

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/raw/hermes-func-calling.json",
                    help="Hermes JSON 输入（数组或 JSONL）")
    ap.add_argument("--out", default="data/toolcall-data.json",
                    help="Alpaca JSONL 输出")
    ap.add_argument("--shuffle", action="store_true", help="打乱输出顺序")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="超过此字符数的样本丢弃（gpt2-medium 上下文 1024 token ≈ 3000-4000 字符，"
                         "保守 2048 字符过滤）")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多保留 N 条（0=全部）")
    ap.add_argument("--stats", action="store_true", help="仅打印统计信息")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_file():
        print(f"[错误] 找不到输入: {src}", file=sys.stderr)
        sys.exit(1)

    n_in = 0
    n_out = 0
    n_too_long = 0
    n_no_tools = 0
    expanded: list[dict] = []
    # 自动判断格式：JSON 数组 / JSONL
    with src.open("r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        print(f"[错误] 文件为空: {src}", file=sys.stderr)
        sys.exit(1)
    if raw[0] == "[":
        # JSON 数组
        try:
            samples = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[错误] JSON 解析失败: {e}", file=sys.stderr)
            sys.exit(1)
        it = enumerate(samples, 1)
    else:
        # JSONL
        samples = []
        for ln_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] 第 {ln_no} 行 JSON 解析失败: {e}", file=sys.stderr)
        it = enumerate(samples, 1)

    for n, sample in it:
        n_in += 1
        ex = expand_one_sample(sample, fallback_system=None)
        if not ex:
            convs = sample.get("conversations") or sample.get("messages") or []
            has_tools = any(
                TOOLS_BLOCK_RE.search(c.get("value", c.get("content", "")))
                for c in convs
                if (c.get("from") or c.get("role", "")).lower() in ("system", "system")
            )
            if not has_tools:
                n_no_tools += 1
            continue
        for e in ex:
            if len(e["instruction"]) > args.max_len:
                n_too_long += 1
                continue
            expanded.append(e)

    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(expanded)
    if args.limit > 0:
        expanded = expanded[:args.limit]
    n_out = len(expanded)

    # 统计
    turns_dist: dict[int, int] = {}
    for e in expanded:
        t = e["_meta"]["turns"]
        turns_dist[t] = turns_dist.get(t, 0) + 1
    avg_instr = sum(len(e["instruction"]) for e in expanded) / max(1, n_out)
    avg_out = sum(len(e["output"]) for e in expanded) / max(1, n_out)

    print(f"--- convert_data 统计 ---")
    print(f"输入:       {n_in} 条对话")
    print(f"无工具块:   {n_no_tools} 条（跳过）")
    print(f"展开后:     {n_out} 条训练样本")
    print(f"超 max-len: {n_too_long} 条（丢弃）")
    print(f"平均 instruction 长度: {avg_instr:.0f} 字符")
    print(f"平均 output 长度:     {avg_out:.0f} 字符")
    print(f"turn 分布: {dict(sorted(turns_dist.items()))}")

    if args.stats:
        return

    # 去掉 _meta 后写出（Alpaca 标准 3 字段）
    out.parent.mkdir(parents=True, exist_ok=True)
    # gpt-sft/core/gpt_sft_core.cpp:load_instruction_data 用 nlohmann::json::operator>>
    # 期望整个文件是 JSON 数组（不是 JSONL）。
    arr = [
        {
            "instruction": e["instruction"],
            "input": e["input"],
            "output": e["output"],
        }
        for e in expanded
    ]
    with out.open("w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False)
    print(f"写出: {out}  ({out.stat().st_size:,} 字节, JSON 数组)")


if __name__ == "__main__":
    main()