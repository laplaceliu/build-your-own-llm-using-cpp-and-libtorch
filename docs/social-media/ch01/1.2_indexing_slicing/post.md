# 1.2 索引 + 切片：避免 80% 的维度错误

> 第 1 章第 2 帖｜阅读 5 分钟｜口播 3 分钟
> 平台：★ 知乎

---

## 钩子

C++ 张量索引最常踩的坑：
**`int` 类型不匹配，编译直接报错。**

---

## 三种索引方式

```cpp
auto x = torch::randn({4, 5, 6});

// 方式 1：连续 []
auto a = x[0];              // shape=[5, 6]
auto b = x[0][1];           // shape=[6]

// 方式 2：链式 [i][j]
auto c = x[0][1][2];        // 单个 scalar

// 方式 3：tensor.index({...})
auto d = x.index({0, 1, 2});// 等价于 x[0][1][2]
```

**推荐**：复杂切片用 `index({...})`，一次到位。

---

## 切片

```cpp
auto x = torch::arange(0, 12).view({3, 4});
// [[0, 1, 2, 3],
//  [4, 5, 6, 7],
//  [8, 9, 10, 11]]

// Python 风格：x[0:2, 1:3]
auto slice = x.slice(0, 0, 2).slice(1, 1, 3);
// [[1, 2],
//  [5, 6]]

// 步长：x[:, ::2]
auto step = x.slice(1, 0, 4, 2);
// [[0, 2],
//  [4, 6],
//  [8, 10]]
```

---

## 整数索引必须用 int64_t

**这是 C++ 特有坑**：

```cpp
int i = 0;
x[i];              // ❌ 编译错误！
// error: incompatible integer to integer conversion

int64_t i = 0;
x[i];              // ✅ OK
```

**原因**：Torch C++ API 用 `int64_t`（= `long long`），而 C++ 默认 `int` 是 32 位。

**解决**：
```cpp
// 写法 1：显式类型
int64_t i = 0;

// 写法 2：用 auto
auto i = int64_t{0};

// 写法 3：用 size_t（部分场景可）
size_t i = 0;  // 但 torch::index 还是需要 int64_t
```

---

## 高级：布尔索引 + where

```cpp
auto x = torch::randn({5});
auto mask = x > 0;
auto positives = x.masked_select(mask);

auto y = torch::where(x > 0, x, -x);  // 绝对值
```

---

## 维度变换速查

| 操作 | C++ | 等价 Python |
|---|---|---|
| Reshape | `x.view({2, 3})` | `x.view(2, 3)` |
| 维度换位 | `x.permute({2, 0, 1})` | `x.permute(2, 0, 1)` |
| 加维 | `x.unsqueeze(0)` | `x.unsqueeze(0)` |
| 压维 | `x.squeeze()` | `x.squeeze()` |

---

## 下一步

**1.3 GPU 加速**：一行代码快 30 倍

---

## 互动

你踩过 `int` vs `int64_t` 的坑吗？
还有什么 C++ 张量操作觉得反直觉？评论区留言。
