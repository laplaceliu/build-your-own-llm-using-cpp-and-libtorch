# X.3 gpt-sft 框架

> 时长：4 分钟｜平台：★ 知乎

## 钩子
训练好的模型怎么用？CLI + HTTP 一行启动

## 要点
1. core 静态库：3 个项目共享
2. sft_train CLI：训练入口
3. sft_serve HTTP：OpenAI 兼容协议
4. 单二进制部署：4.2 MB + 1.4 GB 模型

## 视觉
- 架构图：core / train / serve 三模块
- curl 调用截图

## 素材
- gpt-sft/README.md
- gpt-sft/serve/src/sft_serve.cpp
