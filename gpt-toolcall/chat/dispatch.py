#!/usr/bin/env python3
"""dispatch.py — 多轮 Function Calling dispatcher

调用链：
    1. 渲染 prompt（system + tools + user + 历史 turns）
    2. 写 prompt 到临时文件
    3. 调 tool_chat --json --prompt-file --stop-on "</tool_call>" "<tool_response>"
    4. 解析 JSON 输出 {"text": ..., "stopped_on": ...}
    5. 如果 stopped_on 是 </tool_call> → 解析 tool_call JSON → 真执行 → 拼下一轮
    6. 如果 stopped_on 是 <tool_response> → 异常，模型生成越界（不该发生）
    7. 如果 stopped_on 是 max_new_tokens → 截断警告
    8. 循环直到得到自然语言回答（无 tool_call），或达到 max_turns 上限

工具注册：
    - 内置工具用 @register_tool 装饰器
    - 真实场景应换成 HTTP API / DB / 文件操作
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))
import chat_template as ct  # noqa: E402

# ----------------- 工具注册表 -----------------
TOOLS: dict[str, Callable] = {}
TOOL_DEFS: list[dict] = []


def register_tool(name: str, description: str, parameters: dict):
    """装饰器：把函数注册为可调用的 tool。"""
    def deco(fn):
        TOOLS[name] = fn
        TOOL_DEFS.append({"name": name, "description": description, "parameters": parameters})
        return fn
    return deco


# ----------------- 内置演示工具 -----------------
@register_tool(
    name="get_weather",
    description="获取某城市某日期的天气。",
    parameters={"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名，例如 '北京'"},
        "date": {"type": "string", "description": "ISO 日期，例如 '2026-08-20'，缺省今天"},
        "timezone": {"type": "string", "description": "时区，例如 'Asia/Shanghai'（与模型训练时对齐的可选参数）"},
    }, "required": ["city"]},
)
def get_weather(city: str, date: str | None = None, timezone: str | None = None) -> str:
    # 演示版：硬编码结果。真实场景应调真实 API。
    fake = {
        ("北京", "2026-08-20"): {"temperature": 28, "condition": "多云", "humidity": 65},
        ("上海", "2026-08-20"): {"temperature": 33, "condition": "晴",   "humidity": 78},
        ("广州", "2026-08-20"): {"temperature": 35, "condition": "雷阵雨", "humidity": 82},
    }
    key = (city, date or "2026-08-20")
    if key in fake:
        return json.dumps(fake[key], ensure_ascii=False)
    return json.dumps({"temperature": 25, "condition": "未知", "note": f"演示数据无 {city}/{date}"}, ensure_ascii=False)


@register_tool(
    name="get_time",
    description="返回当前时间（演示）。",
    parameters={"type": "object", "properties": {"timezone": {"type": "string", "description": "时区，例如 'Asia/Shanghai'，缺省 UTC"}}},
)
def get_time(timezone: str = "UTC") -> str:
    return json.dumps({"timezone": timezone, "iso": "2026-08-20T18:30:00+08:00", "note": "演示数据"}, ensure_ascii=False)


@register_tool(
    name="calc",
    description="计算数学表达式（演示，仅支持 +-*/ 与括号）。",
    parameters={"type": "object", "properties": {"expr": {"type": "string", "description": "数学表达式，例如 '(1+2)*3'"}}, "required": ["expr"]},
)
def calc(expr: str) -> str:
    # 简单沙箱：只允许数字、运算符、括号、小数点、空白
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return json.dumps({"error": "非法字符，仅支持 +-*/() 和数字"}, ensure_ascii=False)
    try:
        return json.dumps({"expr": expr, "result": eval(expr)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ----------------- 推理 worker 调用 -----------------
def call_worker(binary: str, prompt: str, model: str, size: str,
                max_new: int = 256, no_cuda: bool = False,
                stop_on: list[str] | None = None) -> dict:
    """调一次 tool_chat 二进制，返回 JSON 解析结果。"""
    stop_on = stop_on or [ct.TOOL_CALL_CLOSE, ct.TOOL_RESPONSE_OPEN]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(prompt)
        prompt_path = f.name
    try:
        cmd = [
            binary,
            "--model", model,
            "--size", size,
            "--prompt-file", prompt_path,
            "--max-new", str(max_new),
            "--json",
        ]
        if no_cuda:
            cmd.append("--no-cuda")
        for s in stop_on:
            cmd.extend(["--stop-on", s])
        env = os.environ.copy()
        # 让 worker 找得到 libtorch
        torch_root = os.environ.get("TORCH_ROOT", "/opt/libtorch")
        env["LD_LIBRARY_PATH"] = f"{torch_root}/lib:" + env.get("LD_LIBRARY_PATH", "")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"tool_chat 退出码 {proc.returncode}\nstderr: {proc.stderr}")
        return json.loads(proc.stdout)
    finally:
        try:
            os.unlink(prompt_path)
        except FileNotFoundError:
            pass


# ----------------- 多轮 loop -----------------
TOOL_CALL_RE = re.compile(
    re.escape(ct.TOOL_CALL_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(ct.TOOL_CALL_CLOSE),
    re.DOTALL,
)


def extract_tool_call(text: str) -> dict | None:
    """从模型输出提取首个 <tool_call>{...}</tool_call>。返回 {name, arguments} 或 None。"""
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return obj
        return None
    except json.JSONDecodeError:
        return None


def run_loop(query: str, binary: str, model: str, size: str,
             max_turns: int = 5, max_new: int = 256, no_cuda: bool = False,
             tools: list[dict] | None = None, verbose: bool = True) -> str:
    """多轮 dispatcher 主入口。返回最终自然语言回答。"""
    tools = tools if tools is not None else TOOL_DEFS
    messages = [{"role": "user", "content": query}]

    for turn in range(max_turns):
        prompt = ct.render_full_prompt(messages, tools=tools)
        if verbose:
            print(f"\n[turn {turn}] prompt ({len(prompt)} chars):\n---\n{prompt}\n---")

        # 单轮生成：max_new 给 512，trigger </tool_call> 时立即停止
        out = call_worker(binary, prompt, model, size,
                          max_new=max_new, no_cuda=no_cuda)
        text = out["text"]
        stopped = out["stopped_on"]
        if verbose:
            print(f"[turn {turn}] generated ({len(text)} chars, stopped_on={stopped!r}):\n{text}\n")

        call = extract_tool_call(text)
        if call is None:
            # 没有 tool_call → 这是最终自然语言回答
            return text

        name = call["name"]
        args = call.get("arguments", {})
        if name not in TOOLS:
            result = json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        else:
            try:
                result = TOOLS[name](**args)
            except Exception as e:
                result = json.dumps({"error": str(e)}, ensure_ascii=False)

        if verbose:
            print(f"[turn {turn}] tool '{name}({args})' -> {result}")

        # 拼下一轮：assistant 说 tool_call，tool 给结果
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "tool", "content": result})

    return text  # 截断时的最后一次回答


# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Function Calling dispatcher")
    ap.add_argument("--binary", required=True, help="tool_chat 二进制路径")
    ap.add_argument("--model", required=True, help="SFT .pth 模型路径")
    ap.add_argument("--size", default="medium", help="sm | medium")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--query", help="单条 query；缺省进入 REPL")
    ap.add_argument("--tools", help="只启用指定工具（逗号分隔）；默认全启用")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.tools:
        wanted = set(args.tools.split(","))
        enabled = [t for t in TOOL_DEFS if t["name"] in wanted]
    else:
        enabled = TOOL_DEFS

    if args.query:
        ans = run_loop(args.query, args.binary, args.model, args.size,
                       max_turns=args.max_turns, max_new=args.max_new,
                       no_cuda=args.no_cuda, tools=enabled, verbose=not args.quiet)
        print(f"\n>>> ANSWER:\n{ans}\n")
        return

    # REPL
    print(f"=== gpt-toolcall dispatcher REPL ===")
    print(f"模型: {args.model} ({args.size})")
    print(f"可用工具: {[t['name'] for t in enabled]}")
    print("输入 quit 退出。\n")
    while True:
        try:
            q = input("QUERY> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q in ("quit", "exit"):
            break
        ans = run_loop(q, args.binary, args.model, args.size,
                       max_turns=args.max_turns, max_new=args.max_new,
                       no_cuda=args.no_cuda, tools=enabled, verbose=False)
        print(f"\n>>> ANSWER:\n{ans}\n")


if __name__ == "__main__":
    main()