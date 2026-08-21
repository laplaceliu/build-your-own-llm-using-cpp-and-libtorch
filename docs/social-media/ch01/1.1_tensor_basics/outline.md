# 1.1 张量的本质：多维数组 + 自动微分

> 时长：3 分钟｜平台：★ 知乎 + ★★ 小红书
> 钩子：「你以为 Tensor 是个张量？它其实是数据块 + 元信息」

## 要点
1. Tensor = data + sizes + dtype + device + requires_grad
2. 6 种创建方式：tensor / zeros / ones / arange / randn / from_blob
3. 必备属性：sizes / dtype / device / numel / dim

## 视觉
- 张量内存布局图（3D → 1D 连续）
- 6 种创建代码截图
- 属性查询截图

## 素材
- `chapters/chapter01_hello_torch/main.cpp` 1-40 行
