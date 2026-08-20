#!/usr/bin/env python3
"""
gpt-bash/eval/bash_eval.py — nl2bash 评测脚本。

对每个测试样本 (nl, gold_cmd):
  1) 调用 bash_chat 二进制生成预测命令
  2) 计算指标：
     EM           : 预测命令 === gold_cmd (去空白)
     AST-match    : 若安装 bashlex, 把两边的 bash AST 规范化后比较
     Safety pass  : 预测命令通过 --exec 模式下的白名单检查
     Exec-safe    : 在受控 sandbox（read-only env、timeout 2s、忽略 stderr）下能跑通而不出错

输出到 JSON：
  metrics = {"em": ..., "ast_match": ..., "safety_pass": ..., "exec_safe": ..., "n": ...}
  preds   = [{"nl":..., "gold":..., "pred":..., "rc":..., "stdout":...}, ...]

用法：
  # 1) 单模型评估
  python bash_eval.py \
      --data gpt-bash/data/bash-instruction-data.json \
      --model gpt-bash/data/bash-sft.pth \
      --size medium \
      --n 200 \
      --out gpt-bash/eval/result-medium.json

  # 2) 多模型比较
  for m in small medium; do
      python bash_eval.py --model <$m.pth> --size $m --n 200 \
          --out gpt-bash/eval/result-$m.json
  done
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# 默认安全白名单 —— 与 bash_chat.cpp 中保持一致
SAFE_ALLOWLIST = {
    "ls", "echo", "printf", "cat", "head", "tail", "wc", "tree", "stat",
    "date", "cal", "whoami", "pwd", "hostname", "uname", "df", "du",
    "free", "uptime", "id", "groups", "tty", "whereis", "which", "type",
    "file", "test", "[", "true", "false", "yes", "seq", "basename",
    "dirname", "readlink", "realpath", "env", "printenv", "set",
    "find", "grep", "egrep", "fgrep", "rgrep", "ag", "rg", "lsattr",
    "diff", "comm", "uniq", "sort", "shuf", "tac", "rev",
    "cut", "tr", "expand", "unexpand", "fold", "fmt", "nl", "od",
    "hexdump", "xxd", "strings",
    "awk", "gawk", "sed", "perl", "xargs", "tee",
    "touch", "mkdir", "ln",
}
BANNED = {
    "rm", "mv", "cp", "dd", "chmod", "chown", "chgrp", "kill", "killall",
    "pkill", "mount", "umount", "sudo", "su", "reboot", "shutdown",
    "halt", "poweroff", "init", "mkfs", "fdisk", "parted",
    "iptables", "firewall-cmd", "systemctl", "service", "useradd",
    "userdel", "passwd", "curl", "wget", "nc", "ncat", "ssh", "scp",
    "rsync", "crontab", "at", "bash", "sh", "zsh", "fish", "python",
    "python3", "perl", "ruby", "eval", "exec", "source",
}


def first_token(cmd: str) -> str:
    """取命令行首个 token 的 basename（不含路径前缀）。"""
    s = cmd.lstrip()
    if not s:
        return ""
    first = s.split(maxsplit=1)[0]
    base = first.rsplit("/", 1)[-1]
    if base.startswith("[") or base.startswith("("):
        return ""
    return base


def normalize(text: str) -> str:
    """轻量规范化：去多余空白。"""
    return re.sub(r"\s+", " ", text.strip())


def is_safe(cmd: str) -> bool:
    tok = first_token(cmd)
    if not tok:
        return False
    if tok in BANNED:
        return False
    return tok in SAFE_ALLOWLIST


def ast_match(gold: str, pred: str) -> bool | None:
    """若 bashlex 可用，AST 节点集合完全一致则 True。不可用返回 None。"""
    try:
        import bashlex  # type: ignore
    except Exception:
        return None
    try:
        g = bashlex.parse(gold)
        p = bashlex.parse(pred)
        return normalize(str(g)) == normalize(str(p))
    except Exception:
        return False


def exec_safe(cmd: str, timeout: int = 2) -> tuple[int, str]:
    """在只读 PATH 下快速试跑 2s，返回 (returncode, stdout)。"""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C",
    }
    try:
        cp = subprocess.run(
            cmd, shell=True, executable="/bin/bash",
            env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        return cp.returncode, cp.stdout
    except subprocess.TimeoutExpired:
        return 124, ""
    except Exception as e:
        return -1, str(e)


def call_chat(binary: Path, model: Path, size: str, nl: str,
              max_new: int = 64, timeout: int = 60) -> str:
    """调用 bash_chat 单条指令模式生成命令。"""
    cp = subprocess.run(
        [str(binary), "--model", str(model), "--size", size,
         "--max-new", str(max_new),
         "--no-cuda",  # 默认 CPU 评测，避免与训练争夺显存
         kDefaultInstruction, nl],
        capture_output=True, text=True, timeout=timeout,
    )
    return extract_pred(cp.stdout)


def extract_pred(stdout: str) -> str:
    """从 bash_chat 输出中提取 '$ ... ' 后面那一行。"""
    out = []
    for ln in stdout.splitlines():
        if ln.startswith("$ "):
            out.append(ln[2:])
    return "\n".join(out).strip() if out else ""


kDefaultInstruction = "Translate the following natural language request into a bash one-liner."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(
        Path(__file__).resolve().parent.parent / "data/bash-instruction-data.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large", "xl"])
    ap.add_argument("--binary", default=str(
        Path(__file__).resolve().parent.parent / "build/bash_chat"))
    ap.add_argument("--n", type=int, default=200, help="评测样本数（随机抽）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", required=True, help="输出 JSON 路径（含 metrics + preds）")
    ap.add_argument("--skip-exec", action="store_true",
                    help="跳过 exec_safe（默认会试跑，要更稳可以打开）")
    ap.add_argument("--start", type=int, default=0,
                    help="从测试集第 N 条开始")
    args = ap.parse_args()

    data_path = Path(args.data)
    model = Path(args.model)
    binary = Path(args.binary)
    out_path = Path(args.out)

    if not data_path.exists():
        print(f"[错误] 数据集不存在: {data_path}", file=sys.stderr); return 1
    if not model.exists():
        print(f"[错误] 模型文件不存在: {model}", file=sys.stderr); return 1
    if not binary.exists():
        print(f"[警告] bash_chat 未找到: {binary}（--size 默认 CPU 评测，"
              f"可能未编译或路径错）", file=sys.stderr)

    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not data:
        print("[错误] 数据集为空", file=sys.stderr); return 1

    random.Random(args.seed).shuffle(data)
    data = data[args.start : args.start + args.n]
    print(f"== 评测集大小 {len(data)} 条（seed={args.seed}, start={args.start}）==")

    metrics = {"n": len(data), "em": 0, "ast_match": 0, "ast_match_na": 0,
               "safety_pass": 0, "exec_safe": 0, "skipped": 0}
    preds = []
    t0 = time.time()
    for i, e in enumerate(data):
        nl, gold = e["input"], e["output"]
        try:
            pred = call_chat(binary, model, args.size, nl, args.max_new)
        except Exception as exc:
            print(f"[{i+1}/{len(data)}] 调用 bash_chat 失败：{exc}", file=sys.stderr)
            metrics["skipped"] += 1
            preds.append({"nl": nl, "gold": gold, "pred": "", "rc": -1, "stdout": ""})
            continue

        em = (normalize(pred) == normalize(gold))
        ast = ast_match(gold, pred)
        safe = is_safe(pred)
        rc, stdout = (-1, "")
        if not args.skip_exec and safe:
            rc, stdout = exec_safe(pred, timeout=2)
        exec_ok = (rc == 0)

        metrics["em"]          += int(em)
        metrics["ast_match"]   += int(ast is True)
        metrics["ast_match_na"]+= int(ast is None)
        metrics["safety_pass"] += int(safe)
        metrics["exec_safe"]   += int(exec_ok)

        preds.append({
            "nl": nl, "gold": gold, "pred": pred,
            "em": em, "ast": ast, "safe": safe,
            "rc": rc, "stdout": stdout,
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(data):
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            print(f"  [{i+1}/{len(data)}] em={metrics['em']}/{i+1}, "
                  f"safe={metrics['safety_pass']}/{i+1} "
                  f"({elapsed:.1f}s, {avg:.2f}s/sample)")

    n = max(1, len(data) - metrics["skipped"])
    summary = {
        "n": len(data),
        "skipped": metrics["skipped"],
        "em":          metrics["em"] / n,
        "safety_pass": metrics["safety_pass"] / n,
        "exec_safe":   metrics["exec_safe"] / n,
        "ast_match":   (metrics["ast_match"] /
                        max(1, n - metrics["ast_match_na"])
                        if (n - metrics["ast_match_na"]) > 0 else None),
        "model": str(model),
        "size": args.size,
        "data": str(data_path),
        "n_param": len(data),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"metrics": summary, "preds": preds},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<14} {v*100:.2f}%")
        else:
            print(f"  {k:<14} {v}")
    print(f"\n[完成] 写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
