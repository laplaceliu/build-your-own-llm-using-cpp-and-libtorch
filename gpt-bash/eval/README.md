# gpt-bash/eval 评测

## 指标

| 指标 | 含义 |
|------|------|
| **EM** | 预测命令 === gold 命令（去空白严格相等）。nl2bash 标准指标。 |
| **AST-match** | 若安装 `bashlex`，比较两侧 bash AST 的字符串表示。 |
| **Safety pass** | 预测命令的 `first_token` 在 `SAFE_ALLOWLIST` 中。 |
| **Exec-safe** | 在只读 `PATH`、`HOME=/tmp`、`timeout=2s` 沙箱中执行返回 rc=0。 |

## 用法

```bash
cd gpt-bash/eval
python bash_eval.py \
    --data ../data/bash-instruction-data.json \
    --model ../data/bash-sft.pth \
    --size medium \
    --n 200 \
    --out result-medium.json
```

## 多模型对比

```bash
for m in small medium; do
    python bash_eval.py --model ../data/bash-sft-$m.pth \
        --size $m --n 200 --out result-$m.json
done
```

## 加自定义验证（可选）

如果你对**某一条**测试样本加了 `expected_stdout`，可在 `bash_eval.py`
里把 `exec_safe` 替换为：

```python
assert cp.returncode == 0
assert normalize(cp.stdout) == normalize(expected_stdout)
```

这样评测就和"命令是否真的产生正确输出"挂钩。
