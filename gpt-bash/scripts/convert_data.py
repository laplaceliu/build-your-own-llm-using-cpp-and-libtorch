#!/usr/bin/env python3
"""
gpt-bash/scripts/convert_data.py

把 TellinaTool/nl2bash 的 all.nl + all.cm 转换成 Alpaca JSON，喂给 gpt-sft 的 sft_train。

输出 JSON 形如：
  [
    {
      "instruction": "Translate the following natural language request into a bash one-liner.",
      "input":       "<nl description>",
      "output":      "<bash command>"
    },
    ...
  ]

可选过滤：
  --max-cmd-len   丢弃 bash 命令长度 > N (字节) 的样本，避免超 BPE 上下文
  --max-nl-len    丢弃 nl 长度 > N (字节) 的样本
  --shuffle       随机打乱
  --seed          随机种子（默认 0）
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path


def read_lines(p: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser(description="nl2bash → Alpaca JSON")
    ap.add_argument("--nl",
        default=str(Path(__file__).resolve().parent.parent / "data/raw/all.nl"))
    ap.add_argument("--cm",
        default=str(Path(__file__).resolve().parent.parent / "data/raw/all.cm"))
    ap.add_argument("--out",
        default=str(Path(__file__).resolve().parent.parent / "data/bash-instruction-data.json"))
    ap.add_argument("--max-cmd-len", type=int, default=200,
        help="bash 命令最大字节数（超过则丢弃），默认 200")
    ap.add_argument("--max-nl-len", type=int, default=400,
        help="nl 描述最大字节数，超过则丢弃，默认 400")
    ap.add_argument("--shuffle", action="store_true", help="随机打乱")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
        help="限制最大样本数（用于快速实验），0=不限制")
    ap.add_argument("--instruction-prefix",
        default="Translate the following natural language request into a bash one-liner.")
    args = ap.parse_args()

    nls = read_lines(Path(args.nl))
    cms = read_lines(Path(args.cm))
    if len(nls) != len(cms):
        print(f"[错误] all.nl ({len(nls)}) 与 all.cm ({len(cms)}) 行数不一致",
              file=sys.stderr)
        return 1

    kept: list[dict] = []
    dropped = 0
    for nl, cm in zip(nls, cms):
        nl_s, cm_s = nl.strip(), cm.strip()
        if not nl_s or not cm_s:
            dropped += 1; continue
        if len(nl_s.encode("utf-8")) > args.max_nl_len:
            dropped += 1; continue
        if len(cm_s.encode("utf-8")) > args.max_cmd_len:
            dropped += 1; continue
        # 跳过"危险"命令样本——仅训练数据过滤；eval 时仍靠运行时白名单兜底
        if any(tok in cm_s for tok in ["rm -rf /", ":(){:|:&};:"]):
            dropped += 1; continue
        kept.append({
            "instruction": args.instruction_prefix,
            "input":       nl_s,
            "output":      cm_s,
        })

    if args.shuffle:
        random.Random(args.seed).shuffle(kept)
    if args.limit and args.limit > 0:
        kept = kept[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[完成] {len(kept)} 条样本写入 {out_path}"
          + (f"（丢弃 {dropped}）" if dropped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
