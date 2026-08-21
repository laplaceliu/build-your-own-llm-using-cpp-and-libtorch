// chapter01_hello_torch/main.cpp
//
// 第 1 章：Hello LibTorch —— 后文用到的 LibTorch 前置知识
//
// 本章用一个自包含的程序，把后续章节（ch2～ch7、chD、chE、chF、chG）
// 都会用到的 LibTorch C++ API 全部演示一遍，便于在正式进入后续章节
// 之前先做一次"沙盘演练"。
//
// 演示范围：
//   §A 设备与全局设置（cuda::is_available / Device / 手动 seed）
//   §B 张量工厂函数（tensor / empty / zeros / ones / arange / randn / from_blob）
//   §C 形变、视图与基本属性（sizes / dtype / device / numel / reshape / view / transpose / permute / contiguous）
//   §D 张量运算（matmul / softmax / cat / masked_fill / argmax / topk / mean）
//   §E 自动求导（requires_grad_ / backward / NoGradGuard / 零梯度）
//   §F nn::Module 子类化（register_module / register_parameter / register_buffer / forward / train/eval）
//   §G 已有 nn 模块（Linear / Embedding / Dropout）与 functional
//   §H 优化器（AdamW / zero_grad / step / get_lr / set_lr）
//   §I 模型遍历（parameters / named_parameters / named_modules / ->at<ImplType>(i)）
//   §J 序列化（torch::save / torch::load 整模型）
//   §K 容器与可选类型（c10::optional<int64_t> / TensorOptions）
//
// 本程序独立运行时不依赖任何后续章节；既可作为阅读时的小型 REPL，
// 也可作为后续章节 LibTorch 写法的"约定速查"。

#include <torch/cuda.h>
#include <torch/torch.h>

#include <c10/util/Optional.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// §A 设备与全局设置（仅本翻译单元用的辅助）
// ---------------------------------------------------------------------------

// 选择计算设备：优先 CUDA，否则 CPU（ch1~chG 每个 main.cpp 都有这 4 行）
torch::Device select_device() {
    if (torch::cuda::is_available()) return torch::Device(torch::kCUDA);
    return torch::Device(torch::kCPU);
}

void header(const std::string& s) {
    std::cout << "\n----- " << s << " -----\n";
}

}  // namespace

// ---------------------------------------------------------------------------
// §F nn::Module 子类化 —— 一个最小但完整的"演示模型"
// ---------------------------------------------------------------------------
//
// 设计目标：用尽量短的代码演示
//   * 子类化 torch::nn::Module
//   * 用 register_module / register_parameter / register_buffer 注册成员
//   * forward(x) 书写计算图
//   * model(x) ↔ model.forward(x)
//   * train() / eval()
//   * named_parameters(true) / named_modules()
//   * ->at<ImplType>(i) 取子模块
//
// 注意：DemoModule / StackMLP 必须放在全局命名空间（不能用匿名 namespace），
// 否则 TORCH_MODULE 生成的 ModuleHolder<...> 与 torch::serialize::operator>>
// 的 ADL 关联会断开，导致 `torch::save/load` 编译失败。

// 真实实现类（Torch 中叫 *Impl）—— TORCH_MODULE 包装类环绕此类
class DemoModuleImpl : public torch::nn::Module {
 public:
    DemoModuleImpl(int64_t in_dim, int64_t hidden_dim) {
        // 已有模块：register_module("name", module)
        fc_       = register_module("fc", torch::nn::Linear(in_dim, hidden_dim));
        dropout_  = register_module("drop", torch::nn::Dropout(0.1));

        // 可训练参数：register_parameter("name", Tensor)
        // 注意 requires_grad 默认 true，由优化器自动收集
        scale_ = register_parameter("scale",
            torch::tensor(1.0).reshape({1}));

        // 非可训练常量：register_buffer("name", Tensor)（随 state_dict 移动但不参与梯度）
        bias_ = register_buffer("bias",
            torch::arange(0, hidden_dim, torch::kFloat));
    }

    torch::Tensor forward(torch::Tensor x) {
        // 演示：fc → relu → +scale → masked_fill → mean
        auto h = torch::relu(fc_->forward(x));
        h = h * scale_;                                            // 广播：(B, H) * (1,)
        // masked_fill：把 <= 0 的位置填为 -inf，常用于 attention mask
        h = h.masked_fill(h <= 0, -1e9);
        return h.mean(/*dim=*/-1, /*keepdim=*/true);
    }

    // 拿一个具体子模块的指针（仅用于展示 typed access）
    torch::nn::Linear fc() { return fc_; }

 private:
    torch::nn::Linear fc_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    torch::Tensor scale_;
    torch::Tensor bias_;
};
TORCH_MODULE(DemoModule);   // 生成 DemoModule : ModuleHolder<DemoModuleImpl>

// 一个更真实的"层列表"演示：Sequential
class StackMLPImpl : public torch::nn::Module {
 public:
    explicit StackMLPImpl(int64_t in_dim, int64_t out_dim) {
        layers_ = register_module("layers", torch::nn::Sequential(
            torch::nn::Linear(in_dim, 32),
            torch::nn::ReLU(),
            torch::nn::Linear(32, out_dim)));
    }
    torch::Tensor forward(torch::Tensor x) { return layers_->forward(x); }
 private:
    torch::nn::Sequential layers_{nullptr};
};
TORCH_MODULE(StackMLP);

int main() {
    std::cout << "=== Chapter 01: Hello LibTorch ===\n";
    std::cout << "Torch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR
              << "." << TORCH_VERSION_PATCH << "\n";

    // ----- §A 设备与全局设置 -----
    header("A. Device & Global Settings");
    std::cout << "CUDA available: "
              << (torch::cuda::is_available() ? "yes" : "no") << "\n";
    torch::Device device = select_device();
    std::cout << "Using device: " << device << "\n";
    torch::manual_seed(123);   // 与书中数值保持一致
    std::cout << "manual_seed(123) done on CPU; weights will migrate to device later\n";
    // 教学约束：禁用 TF32，与书中 Python float32 数值对齐
    at::globalContext().setAllowTF32CuBLAS(false);
    std::cout << "TF32 (CuBLAS) disabled to match book floats\n";

    // ----- §B 张量工厂函数 -----
    header("B. Tensor Factories");
    auto x = torch::tensor({{1.0f, 2.0f, 3.0f},
                            {4.0f, 5.0f, 6.0f}});                // 自动推断 (2,3) Float
    auto w = torch::tensor({{1.0f}, {1.0f}, {1.0f}});

    auto empty = torch::empty({2, 3});                            // 不初始化
    auto zeros = torch::zeros({2, 3});                            // 全 0
    auto ones  = torch::ones({2, 3});                             // 全 1
    auto rng   = torch::rand({2, 3});                             // [0,1) 均匀
    auto rgn   = torch::randn({2, 3});                            // 标准正态
    auto arng  = torch::arange(0, 6, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));   // [0..6) kLong
    std::vector<int64_t> ids{101, 102, 103, 104};
    auto tids  = torch::tensor(ids, torch::kLong);                // 字符级 ID → Long Tensor
    auto from_blob_t = torch::from_blob(                          // 复用裸指针内存（仅展示签名）
        const_cast<float*>(x.data_ptr<float>()), {2, 3}, torch::kFloat);
    std::cout << "x  shape=" << x.sizes()    << " dtype=" << x.dtype()
              << " device=" << x.device()    << "\n";
    std::cout << "rng:\n" << rng << "\n";
    (void)empty; (void)zeros; (void)ones; (void)rgn; (void)arng; (void)tids; (void)from_blob_t;

    // ----- §C 形变 / 视图 / 属性 -----
    header("C. Shape, View, Property");
    auto a = torch::arange(12, torch::kFloat).reshape({3, 4});    // (3,4)
    auto b = a.transpose(0, 1);                                   // (4,3)
    auto c = a.permute({1, 0});                                   // (4,3)
    auto d = a.view({-1});                                        // (12,) -1 维度自动推断
    auto e = a.unsqueeze(0).unsqueeze(0);                         // (1,1,3,4)
    auto f = e.squeeze(0).squeeze(0);                             // 回到 (3,4)
    auto g = a.reshape({2, 2, 3});                                // 与 view 类似但更通用
    auto h = a.contiguous();                                      // 强制连续（某些 op 要求）
    std::cout << "a.numel()=" << a.numel() << ", a.contiguous=" << a.is_contiguous() << "\n";
    (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;

    // ----- §D 张量运算 -----
    header("D. Tensor Ops");
    auto y_mat = x.matmul(w);                                     // (2,3)@(3,1)=(2,1)
    auto y_soft = torch::softmax(x, /*dim=*/1);                   // 沿最后一维归一
    auto y_cat  = torch::cat({x, x}, /*dim=*/0);                  // 沿行拼接 → (4,3)
    auto y_mask = x.masked_fill(x > 3.0f, -1e9);                  // >3 的位置填 -inf
    auto y_amax = x.argmax(/*dim=*/1);                            // 每行最大下标
    auto y_topk = std::get<0>(x.topk(/*k=*/2, /*dim=*/1));        // 每行 top-2
    auto y_mean = x.mean();                                        // 全局均值（标量）
    std::cout << "matmul:\n" << y_mat << "\nsoftmax:\n" << y_soft << "\n";
    (void)y_cat; (void)y_mask; (void)y_amax; (void)y_topk; (void)y_mean;

    // ----- §E 自动求导 -----
    header("E. Autograd");
    auto xg = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat).requires_grad_(true);
    auto yg = (xg * xg).sum();                                    // y = sum(x*x)，dy/dx = 2x
    yg.backward();                                                // x.grad = [2, 4, 6]
    std::cout << "y = sum(x*x) ; grad at x: " << xg.grad() << "\n";
    {
        // NoGradGuard：进入推理/评估代码块（chD 测试、chG 推理固定动作）
        torch::NoGradGuard no_grad;
        auto x_eval = x.to(device);
        auto y_eval = x_eval.matmul(w.to(device));
        std::cout << "[no_grad] y on " << y_eval.device() << "\n";
    }

    // ----- §F nn::Module 子类 + §G 已有模块 + §I 模型遍历 -----
    header("F/G/I. nn::Module, nn layers, parameter traversal");
    auto model = DemoModule(/*in_dim=*/3, /*hidden_dim=*/4);
    auto head  = StackMLP(/*in_dim=*/4, /*out_dim=*/2);
    (void)head;   // 仅作演示

    // parameters()  是 optimizer 的入口；返回的是 Generator<Variable>
    int64_t n_params = 0;
    for (const auto& p : model->parameters()) {
        if (p.requires_grad()) n_params += p.numel();
    }
    std::cout << "trainable params (DemoModule) = " << n_params << "\n";

    // named_parameters(true)：递归 + 拿到名字 + 实现冻结/解冻
    std::cout << "named_parameters(true):\n";
    for (const auto& np : model->named_parameters(/*recurse=*/true)) {
        std::cout << "  " << np.key() << "  shape=" << np.value().sizes() << "\n";
    }

    // named_modules()：递归得到 (name, module) 对，常见于权重可视化/统计
    std::cout << "named_modules():\n";
    for (const auto& nm : model->named_modules()) {
        std::cout << "  " << (nm.key().empty() ? "<root>" : nm.key()) << "\n";
    }

    // 拿到具体子模块的接口：直接用 fc_ 是 shared_ptr<LinearImpl>
    std::cout << "fc weight shape: " << model->fc()->weight.sizes() << "\n";

    // ----- Forward & 训练 / 评估模式 -----
    model->to(device);                          // 让参数 + buffer 同步到目标设备
    auto x_dev = torch::randn({2, 3}).to(device);
    model->train();
    auto logits = model->forward(x_dev);
    model->eval();
    {
        torch::NoGradGuard no_grad;
        auto logits_eval = model->forward(x_dev);
        std::cout << "logits eval shape=" << logits_eval.sizes()
                  << " device=" << logits_eval.device() << "\n";
    }

    // ----- §H 优化器 -----
    header("H. Optimizer (AdamW)");
    auto opt = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(1e-3).betas({0.9, 0.95}).weight_decay(0.01));
    {
        // 拿到首个 param_group 的 options，再 downcast 回 AdamWOptions 取具体字段
        auto& opts = static_cast<torch::optim::AdamWOptions&>(
            opt.param_groups().front().options());
        std::cout << "initial lr=" << opts.get_lr()
                  << ", betas=" << std::get<0>(opts.betas())
                  << "," << std::get<1>(opts.betas())
                  << ", wd=" << opts.weight_decay() << "\n";
    }

    // 训练循环 5 行模板（ch5～chE 全书都用）
    model->train();
    opt.zero_grad();                                              // 1. 清零
    auto target = torch::zeros({2, 1}).to(device);
    auto loss   = ((logits - target).pow(2)).mean();              // 2. 计算 MSE loss（手写便于入门）
    loss.backward();                                              // 3. 反向
    opt.step();                                                   // 4. 更新
    std::cout << "one train step loss=" << loss.item<float>() << "\n";

    // 动态调整学习率（ch5 cosine / ch6 / chD）
    opt.param_groups().front().options().set_lr(1e-4);
    std::cout << "after set_lr: "
              << opt.param_groups().front().options().get_lr() << "\n";

    // ----- §J 序列化 -----
    header("J. Serialization (save / load the whole model)");
    {
        torch::NoGradGuard no_grad;
        const std::string path = "/tmp/chapter01_demo_model.pt";
        torch::save(model, path);
        auto model_loaded = DemoModule(/*in_dim=*/3, /*hidden_dim=*/4);
        torch::load(model_loaded, path);
        auto out = model_loaded->forward(x_dev);
        std::cout << "loaded model output shape=" << out.sizes() << "\n";
        std::remove(path.c_str());
    }

    // ----- §K 容器与可选类型 -----
    header("K. c10::optional / TensorOptions");
    c10::optional<int64_t> maybe_topk = c10::nullopt;             // 默认不开启 top_k
    if (maybe_topk.has_value()) {
        std::cout << "top_k = " << maybe_topk.value() << "\n";
    } else {
        std::cout << "top_k disabled (c10::nullopt)\n";
    }

    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat)
                    .device(device)
                    .requires_grad(true);
    auto weight = torch::randn({3, 3}, opts);                      // 全套选项一次生效
    std::cout << "weight device=" << weight.device()
              << " requires_grad=" << weight.requires_grad() << "\n";

    // ----- 收尾 -----
    header("End of Chapter 01 prelude");
    std::cout << "Original first demo  -> x @ w =\n" << y_mat << "\n";
    return 0;
}
