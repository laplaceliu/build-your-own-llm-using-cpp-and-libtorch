# X.1 gpt-bash：自然语言 → Bash 命令

> 扩展项目第 1 帖｜⭐ 实战｜阅读 7 分钟｜口播 4 分钟
> 平台：★ 知乎 + ★★ 小红书

---

## 钩子

**「把当前目录最大的 5 个文件列出来」—— 这种命令，不用记，模型帮你写。**

---

## 项目结构

```
gpt-bash/
├── README.md
├── chat/                 # 推理入口
│   ├── chat.cpp          # C++ 单轮 worker
│   └── dispatcher.py     # Python 多轮 + 沙箱
├── data/
│   └── nl2bash.jsonl     # 12k 样本
├── scripts/
│   ├── train.sh
│   └── serve.sh
└── config/
    └── sandbox.json      # 沙箱白名单
```

---

## 数据集：nl2bash

```
{
  "instruction": "List the top 5 largest files in the current directory",
  "bash": "du -ah . | sort -rh | head -5"
}
{
  "instruction": "Find all files larger than 100MB in /var",
  "bash": "find /var -size +100M -type f"
}
```

**12,000 样本**，覆盖 ls/find/grep/awk/sed 等常用命令。

---

## 训练流程

```bash
cd gpt-bash
HF_MIRROR=hf-mirror ./scripts/download-weights.sh gpt2-small

python -m gpt_sft.train \
    --data data/nl2bash.jsonl \
    --model gpt2-small \
    --epochs 5 \
    --batch-size 8 \
    --lr 5e-5 \
    --output ./checkpoints/gpt2-bash
```

训练输出：
```
epoch 0 loss=2.31
epoch 1 loss=1.42
epoch 2 loss=0.93
epoch 3 loss=0.61
epoch 4 loss=0.42
```

---

## 沙箱：防危险命令

```json
// config/sandbox.json
{
  "allowed_commands": ["ls", "cat", "head", "tail", "grep", "find",
                        "du", "wc", "sort", "uniq", "awk", "sed"],
  "blocked_patterns": [
    "rm\\s+-rf",
    "mkfs",
    "dd\\s+if=",
    ":(){ :|:& };:",
    "sudo"
  ],
  "max_execution_time": 5,
  "working_dir": "/tmp/sandbox"
}
```

```python
def is_safe(cmd: str) -> bool:
    for pattern in sandbox["blocked_patterns"]:
        if re.search(pattern, cmd):
            return False
    
    first_word = cmd.split()[0]
    return first_word in sandbox["allowed_commands"]
```

---

## 使用示例

### CLI 模式

```bash
$ ./build/chat --model ./checkpoints/gpt2-bash
> 列出当前目录下最大的 5 个文件
du -ah . | sort -rh | head -5

> 在 /var/log 找 ERROR 日志
grep -r "ERROR" /var/log/

> 显示当前用户的前 10 个进程
ps aux | head -10
```

### Python 多轮 + 沙箱

```python
from dispatcher import BashChat

chat = BashChat(model_path="./checkpoints/gpt2-bash",
                sandbox="config/sandbox.json")

# 多轮对话
print(chat.ask("列出 home 目录的文件"))
# ls -la /home

print(chat.ask("只显示文件大小"))
# ls -la /home | awk '{print $5, $9}'

# 执行（白名单通过）
result = chat.run("ls -la /home")
# drwxr-xr-x 5 user user 4096 Aug 21 10:00 user1
# drwxr-xr-x 3 user user 4096 Aug 21 09:00 user2
```

---

## 5 组实测

| 指令 | 模型输出 | 是否可执行 |
|---|---|---|
| 列出最大的 5 个文件 | `du -ah . \| sort -rh \| head -5` | ✅ |
| 在 /var 找 100M+ 文件 | `find /var -size +100M -type f` | ✅ |
| 显示系统信息 | `uname -a` | ✅ |
| 删除所有 tmp 文件 | `rm -rf /tmp/*` | 🚫 沙箱拦截 |
| 修改 hosts | `echo "127.0.0.1 x.com" >> /etc/hosts` | 🚫 沙箱拦截 |

**5/5 命令语法正确，4/5 安全范围内，1/5 沙箱正确拦截**。

---

## 性能数据

| 项 | 值 |
|---|---|
| 模型大小 | 498 MB |
| 推理延迟 | ~200 ms / 命令 |
| 训练时长（5 epoch） | ~30 分钟（RTX 4080） |
| 命令语法正确率 | ~85% |
| 命令准确率（语义对） | ~70% |

---

## 下一步

**X.2 gpt-toolcall**：让模型学会调工具

---

## 互动

你最希望 LLM 帮你生成什么类型的命令？
- 文件操作（find/ls/grep）
- Git 操作（commit/branch/diff）
- Docker 操作（ps/logs/exec）
- 其他

评论告诉我。
