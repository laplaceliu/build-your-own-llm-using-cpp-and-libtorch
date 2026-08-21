# 2.4 滑动窗口采样：把长文切成训练样本

> 第 2 章第 4 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

**模型输入长度是固定的，但文本是无限的——怎么切？**

滑动窗口：把长文本切成一段段等长的「窗口」，喂给模型。

---

## 滑动窗口示意

原文（100 tokens）：
```
[0, 1, 2, ..., 99]
```

配置：`max_length=4, stride=4`（无重叠）：
```
Batch 1: [0, 1, 2, 3]   Target: [1, 2, 3, 4]
Batch 2: [4, 5, 6, 7]   Target: [5, 6, 7, 8]
...
Batch 25: [96, 97, 98, 99]  Target: [97, 98, 99, 100]
```

配置：`max_length=4, stride=2`（重叠）：
```
Batch 1: [0, 1, 2, 3]    Target: [1, 2, 3, 4]
Batch 2: [2, 3, 4, 5]    Target: [3, 4, 5, 6]
...
```

**stride 越小 = 重叠越多 = 训练样本越多，但有冗余**。

---

## 输入 vs 目标：右移一位

```cpp
auto input  = torch::tensor({0, 1, 2, 3, 4, 5});  // shape [6]
auto target = torch::tensor({1, 2, 3, 4, 5, 6});  // shape [6]，右移一位
```

**为什么右移一位？**
预测「下一个 token」是语言模型的基本任务。

可视化：
```
输入:  [我, 爱, 吃, 苹, 果]
目标:  [爱, 吃, 苹, 果, 。]
       ↑ 预测目标就是下一位
```

---

## GPTDataLoader 实现

```cpp
class GPTDataLoader {
public:
    GPTDataLoader(const std::vector<int>& tokens,
                  int max_length, int stride, int batch_size)
        : tokens_(tokens), max_length_(max_length),
          stride_(stride), batch_size_(batch_size),
          cursor_(0) {}
    
    // 返回一个 batch 的 (input, target)
    std::pair<torch::Tensor, torch::Tensor> next_batch() {
        std::vector<int> xs, ys;
        for (int b = 0; b < batch_size_; ++b) {
            if (cursor_ + max_length_ + 1 > tokens_.size()) {
                cursor_ = 0;  // 循环
            }
            
            auto chunk = std::vector<int>(
                tokens_.begin() + cursor_,
                tokens_.begin() + cursor_ + max_length_ + 1
            );
            
            for (int i = 0; i < max_length_; ++i) {
                xs.push_back(chunk[i]);
                ys.push_back(chunk[i + 1]);
            }
            cursor_ += stride_;
        }
        
        auto x = torch::tensor(xs).view({batch_size_, max_length_});
        auto y = torch::tensor(ys).view({batch_size_, max_length_});
        return {x, y};
    }
    
private:
    std::vector<int> tokens_;
    int max_length_, stride_, batch_size_, cursor_;
};
```

---

## 关键参数选择

| 参数 | 推荐值 | 影响 |
|---|---|---|
| `max_length` | 256 / 512 / 1024 | 上下文窗口，决定显存 |
| `stride` | = max_length | 不重叠，省内存 |
| `batch_size` | 8 / 16 / 32 | 越大训练越稳，显存越多 |
| `shuffle` | True | 每 epoch 重新随机起点 |

**GPT-2 small 训练**：`max_length=1024, batch_size=8, stride=1024`
**GPT-2 medium 训练**：`max_length=1024, batch_size=4, stride=1024`（显存压力）

---

## 一个常见坑

```cpp
// ❌ 用 chunk 而不是 view → 内存不连续
auto x = torch::from_blob(xs.data(), {batch_size_, max_length_});

// ✅ 用 tensor() 自动拷贝
auto x = torch::tensor(xs).view({batch_size_, max_length_});
```

`from_blob` 只持有指针，后续 xs 析构后 x 变野指针。

---

## 下一步

**2.5 词嵌入 + 位置编码**：让模型理解词序

---

## 互动

你做训练时 `max_length` 设多大？
- 256（显存紧）
- 512（折中）
- 1024（标准）
- 2048+（土豪）

评论告诉我。
