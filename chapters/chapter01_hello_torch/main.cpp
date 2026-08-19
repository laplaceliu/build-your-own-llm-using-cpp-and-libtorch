// chapter01_hello_torch/main.cpp
// 第 1 章：Hello LibTorch —— 张量基础
//
// 目标：
//   1. 验证 LibTorch 环境可编译、可运行
//   2. 认识 torch::Tensor：创建张量、查看形状/数据类型/设备
//   3. 完成第一次矩阵乘法（CUDA 可用时在 GPU 上执行）
#include <torch/cuda.h>
#include <torch/torch.h>

#include <iostream>

// 选择计算设备：优先 CUDA，否则 CPU
torch::Device select_device() {
    if (torch::cuda::is_available()) return torch::Device(torch::kCUDA, 0);
    return torch::Device(torch::kCPU);
}

int main() {
    std::cout << "=== Chapter 01: Hello LibTorch ===\n";

    // LibTorch 版本
    std::cout << "Torch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR
              << "." << TORCH_VERSION_PATCH << "\n";
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "yes" : "no")
              << "\n";

    torch::Device device = select_device();
    std::cout << "Using device: " << device << "\n";

    // 创建两个张量：x 为 2x3，w 为 3x1（迁移到计算设备）
    auto x = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}).to(device);
    auto w = torch::tensor({{1.0f}, {1.0f}, {1.0f}}).to(device);

    std::cout << "x:\n" << x << "\n";
    std::cout << "w:\n" << w << "\n";

    // 矩阵乘法 x @ w -> (2,1)（在 GPU 上执行）
    auto y = x.matmul(w);
    std::cout << "x @ w:\n" << y << "\n";

    // 张量元信息：形状 / 数据类型 / 设备
    std::cout << "shape = " << x.sizes() << ", dtype = " << x.dtype()
              << ", device = " << x.device() << "\n";

    return 0;
}
