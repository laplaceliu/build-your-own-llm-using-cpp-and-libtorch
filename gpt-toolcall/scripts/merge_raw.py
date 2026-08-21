#!/usr/bin/env python3
"""merge_raw.py — 把 data/raw/ 下所有 Hermes JSON 合并为一个 JSON 数组
，供 convert_data.py 一次性解析。
"""
import json
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = RAW / "all-merged.json"

samples = []
for f in sorted(RAW.glob("*.json")):
    if f.name == "all-merged.json":
        continue
    try:
        d = json.load(open(f))
    except json.JSONDecodeError as e:
        print(f"[warn] {f}: {e}", file=sys.stderr)
        continue
    print(f"[merge] {f.name}: {len(d)} samples")
    samples.extend(d)

print(f"[merge] total: {len(samples)} samples → {OUT}")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False)
print(f"[merge] {OUT.stat().st_size:,} bytes")