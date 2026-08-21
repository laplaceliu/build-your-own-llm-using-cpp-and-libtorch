# 5.3 训练循环

> 时长：4 分钟｜平台：★ 知乎

## 钩子
训练循环只有 5 步，CPU 跑 8 分钟，GPU 跑 12 秒

## 要点
1. 5 步：forward → loss → backward → step → zero_grad
2. 完整可运行代码
3. 实测：CPU 8min vs GPU 12s（40x）
4. 4 个易错点：zero_grad / train mode / device / .item()

## 视觉
- 训练循环伪代码
- 终端输出：epoch 1/10 loss=11.2 ...
- GPU vs CPU 柱状图

## 素材
- chapters/chapter05_pretraining/src/main.cpp
