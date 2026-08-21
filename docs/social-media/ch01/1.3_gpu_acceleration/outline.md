# 1.3 GPU 加速

> 时长：2 分钟｜平台：★★ 小红书

## 钩子
`.to(torch::kCUDA)` 这一行，帮你省 30 倍时间

## 要点
1. `torch::cuda::is_available()` 检测
2. 三步切换：检测 → 切模型 → 切数据
3. 实测：本项目预训练 8min→12s
4. 4 条原则：同设备 / 不频繁 / 返回新对象 / 存盘切 CPU

## 视觉
- 终端 cuda:0 输出截图
- CPU vs GPU 柱状图
- nvidia-smi 截图

## 素材
- README.md 「GPU 支持」段
