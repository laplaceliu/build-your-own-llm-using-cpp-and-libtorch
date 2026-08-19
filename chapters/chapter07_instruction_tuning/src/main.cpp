// main.cpp
// 第 7 章：通过微调遵循人类指令（C++ + LibTorch 实现）
//
// 按书中步骤演示，中间结果全部打印：
//   7.2  下载/格式化/划分指令数据集（1100 条 -> 935/55/110）
//   7.3  自定义聚合函数 custom_collate_fn（填充 + 目标左移 + -100 掩码）
//   7.4  指令数据加载器
//   7.5  加载 GPT-2 medium (355M) 预训练权重 + 微调前基准回复
//   7.6  指令微调（AdamW 2 轮）
//   7.7  抽取并保存测试集模型回复（instruction-data-with-response.json）
//   7.8  用 Ollama (Llama3) 自动评分（可选，需 ollama + llama3）
//
// 用法：
//   ./chapter07_instruction_tuning [data_dir] [epochs]
#include <torch/cuda.h>
#include <torch/torch.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bpe_tokenizer.h"
#include "gpt.h"
#include "instruction.h"
#include "safetensors.h"
#include "training.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif
#ifndef TOKENIZER_DIR
#define TOKENIZER_DIR "../chapter02_text_data/data"
#endif

namespace {

void section(const std::string& title) {
    std::cout << "\n========================================================\n"
              << title << "\n"
              << "========================================================\n";
}

// ---- 读取 JSON 指令数据集 ----
std::vector<ch7::InstructionEntry> read_instruction_json(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("无法打开数据集: " + path);
    nlohmann::json j;
    ifs >> j;
    std::vector<ch7::InstructionEntry> data;
    for (const auto& e : j) {
        data.push_back({e["instruction"].get<std::string>(),
                        e["input"].get<std::string>(),
                        e["output"].get<std::string>()});
    }
    return data;
}

// ---- 保存测试集回复 JSON ----
void save_response_json(const std::string& path,
                        const std::vector<ch7::InstructionEntry>& test_data) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& e : test_data) {
        arr.push_back({{"instruction", e.instruction}, {"input", e.input},
                       {"output", e.output}, {"model_response", e.model_response}});
    }
    std::ofstream ofs(path);
    ofs << arr.dump(4);
}

// ---- 7.3 交叉熵 ignore_index 演示 ----
void demo_cross_entropy() {
    section("7.3b 交叉熵损失与 ignore_index=-100");
    auto logits_1 = torch::tensor({{-1.0f, 1.0f}, {-0.5f, 1.5f}});
    auto targets_1 = torch::tensor({0L, 1L});
    auto loss_1 = torch::nn::functional::cross_entropy(logits_1, targets_1);
    std::cout << "loss_1 = " << loss_1 << "\n";

    auto logits_2 = torch::tensor({{-1.0f, 1.0f}, {-0.5f, 1.5f}, {-0.5f, 1.5f}});
    auto targets_2 = torch::tensor({0L, 1L, 1L});
    auto loss_2 = torch::nn::functional::cross_entropy(logits_2, targets_2);
    std::cout << "loss_2 = " << loss_2 << "\n";

    auto targets_3 = torch::tensor({0L, 1L, -100L});
    auto loss_3 = torch::nn::functional::cross_entropy(logits_2, targets_3);
    std::cout << "loss_3 (ignore -100) = " << loss_3 << "\n";
    std::cout << "loss_1 == loss_3: "
              << (std::fabs(loss_1.item<double>() - loss_3.item<double>()) < 1e-6 ? "yes" : "no")
              << "\n";
}

// ---- 5.5 权重加载（参数化支持 gpt2-medium）----
void load_gpt2_weights(ch4::GPTModel& gpt, const std::string& st_path) {
    auto st = ch5::load_safetensors(st_path);
    int64_t n_layers = gpt->trf_blocks->size();
    int64_t emb = gpt->tok_emb->weight.size(1);

    auto map_key = [](const std::string& k) -> std::string {
        if (k == "wte") return "wte.weight";
        if (k == "wpe") return "wpe.weight";
        if (k == "g") return "ln_f.weight";
        if (k == "b") return "ln_f.bias";
        if (k.rfind("blocks.", 0) == 0) {
            std::string rest = k.substr(7);
            size_t dot = rest.find('.');
            std::string idx = rest.substr(0, dot);
            std::string tail = rest.substr(dot + 1);
            if (tail == "ln_1.g") return "h." + idx + ".ln_1.weight";
            if (tail == "ln_1.b") return "h." + idx + ".ln_1.bias";
            if (tail == "ln_2.g") return "h." + idx + ".ln_2.weight";
            if (tail == "ln_2.b") return "h." + idx + ".ln_2.bias";
            if (tail == "attn.c_attn.w") return "h." + idx + ".attn.c_attn.weight";
            if (tail == "attn.c_attn.b") return "h." + idx + ".attn.c_attn.bias";
            if (tail == "attn.c_proj.w") return "h." + idx + ".attn.c_proj.weight";
            if (tail == "attn.c_proj.b") return "h." + idx + ".attn.c_proj.bias";
            if (tail == "mlp.c_fc.w") return "h." + idx + ".mlp.c_fc.weight";
            if (tail == "mlp.c_fc.b") return "h." + idx + ".mlp.c_fc.bias";
            if (tail == "mlp.c_proj.w") return "h." + idx + ".mlp.c_proj.weight";
            if (tail == "mlp.c_proj.b") return "h." + idx + ".mlp.c_proj.bias";
        }
        return k;
    };
    auto assign = [](torch::Tensor left, const torch::Tensor& right) {
        if (left.sizes() != right.sizes())
            throw std::runtime_error("Shape mismatch: " + std::to_string(left.numel()) +
                                     " vs " + std::to_string(right.numel()));
        { torch::NoGradGuard g; left.copy_(right); }
        return left;
    };
    auto params = [&](const std::string& name) -> const torch::Tensor& {
        return st.at(map_key(name));
    };

    gpt->pos_emb->weight = assign(gpt->pos_emb->weight, params("wpe"));
    gpt->tok_emb->weight = assign(gpt->tok_emb->weight, params("wte"));

    for (int64_t b = 0; b < n_layers; ++b) {
        auto& block = gpt->trf_blocks->at<ch4::TransformerBlockImpl>(b);
        auto c_attn_w = params("blocks." + std::to_string(b) + ".attn.c_attn.w");
        auto split = c_attn_w.split(emb, 1);
        block.att->W_query->weight = assign(block.att->W_query->weight, split[0].t().contiguous());
        block.att->W_key->weight = assign(block.att->W_key->weight, split[1].t().contiguous());
        block.att->W_value->weight = assign(block.att->W_value->weight, split[2].t().contiguous());
        auto c_attn_b = params("blocks." + std::to_string(b) + ".attn.c_attn.b");
        auto split_b = c_attn_b.split(emb, 0);
        block.att->W_query->bias = assign(block.att->W_query->bias, split_b[0]);
        block.att->W_key->bias = assign(block.att->W_key->bias, split_b[1]);
        block.att->W_value->bias = assign(block.att->W_value->bias, split_b[2]);
        block.att->out_proj->weight = assign(
            block.att->out_proj->weight,
            params("blocks." + std::to_string(b) + ".attn.c_proj.w").t().contiguous());
        block.att->out_proj->bias = assign(
            block.att->out_proj->bias,
            params("blocks." + std::to_string(b) + ".attn.c_proj.b"));
        auto& ff0 = block.ff->layers->at<torch::nn::LinearImpl>(0);
        ff0.weight = assign(ff0.weight,
                            params("blocks." + std::to_string(b) + ".mlp.c_fc.w").t().contiguous());
        ff0.bias = assign(ff0.bias, params("blocks." + std::to_string(b) + ".mlp.c_fc.b"));
        auto& ff2 = block.ff->layers->at<torch::nn::LinearImpl>(2);
        ff2.weight = assign(ff2.weight,
                            params("blocks." + std::to_string(b) + ".mlp.c_proj.w").t().contiguous());
        ff2.bias = assign(ff2.bias, params("blocks." + std::to_string(b) + ".mlp.c_proj.b"));
        block.norm1->scale_ = assign(block.norm1->scale_,
                                     params("blocks." + std::to_string(b) + ".ln_1.g"));
        block.norm1->shift_ = assign(block.norm1->shift_,
                                     params("blocks." + std::to_string(b) + ".ln_1.b"));
        block.norm2->scale_ = assign(block.norm2->scale_,
                                     params("blocks." + std::to_string(b) + ".ln_2.g"));
        block.norm2->shift_ = assign(block.norm2->shift_,
                                     params("blocks." + std::to_string(b) + ".ln_2.b"));
    }
    gpt->final_norm->scale_ = assign(gpt->final_norm->scale_, params("g"));
    gpt->final_norm->shift_ = assign(gpt->final_norm->shift_, params("b"));
    gpt->out_head->weight = assign(gpt->out_head->weight, params("wte"));
}

// ---- 指令微调的损失计算（ignore_index=-100）----
torch::Tensor calc_loss_batch(const torch::Tensor& input_batch,
                              const torch::Tensor& target_batch, ch4::GPTModel& model) {
    auto logits = model->forward(input_batch);
    return torch::nn::functional::cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(),
        torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));
}

double calc_loss_loader(ch4::GPTModel& model,
                        const std::vector<ch7::InstructionBatch>& loader,
                        c10::optional<int64_t> num_batches = c10::nullopt) {
    if (loader.empty()) return std::numeric_limits<double>::quiet_NaN();
    int64_t n = num_batches.has_value()
                    ? std::min(num_batches.value(), static_cast<int64_t>(loader.size()))
                    : static_cast<int64_t>(loader.size());
    double total = 0.0;
    torch::NoGradGuard g;
    model->eval();
    for (int64_t i = 0; i < n; ++i)
        total += calc_loss_batch(loader[i].inputs, loader[i].targets, model).item<double>();
    model->train();
    return total / n;
}

std::pair<double, double> evaluate_model(ch4::GPTModel& model,
                                         const std::vector<ch7::InstructionBatch>& train_loader,
                                         const std::vector<ch7::InstructionBatch>& val_loader,
                                         int64_t eval_iter) {
    return {calc_loss_loader(model, train_loader, eval_iter),
            calc_loss_loader(model, val_loader, eval_iter)};
}

// ---- 指令微调训练循环（代码清单 7-8，步数 6 位对齐书）----
void train_model_simple(ch4::GPTModel& model,
                        const std::vector<ch7::InstructionBatch>& train_loader,
                        const std::vector<ch7::InstructionBatch>& val_loader,
                        torch::optim::AdamW& optimizer, int64_t num_epochs,
                        int64_t eval_freq, int64_t eval_iter,
                        const std::string& start_context, ch2::BpeTokenizer& tokenizer) {
    int64_t global_step = -1;
    torch::Device dev = model->tok_emb->weight.device();
    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        for (const auto& batch : train_loader) {
            optimizer.zero_grad();
            auto loss = calc_loss_batch(batch.inputs, batch.targets, model);
            loss.backward();
            optimizer.step();
            global_step += 1;
            if (global_step % eval_freq == 0) {
                auto [tl, vl] = evaluate_model(model, train_loader, val_loader, eval_iter);
                std::cout << "Ep " << (epoch + 1) << " (Step " << std::setw(6) << std::setfill('0')
                          << global_step << "): Train loss " << tl << ", Val loss " << vl << "\n";
            }
        }
        // 每轮结束：在验证集首条指令上生成回复（与书一致）
        std::cout << "Below is an instruction that describes a task. Write a response "
                  << "that appropriately completes the request.";
        std::cout << " ### Instruction: Convert the active sentence to passive: "
                  << "'The chef cooks the meal every day.'\n";
        model->eval();
        auto encoded = ch5::text_to_token_ids(start_context, tokenizer).to(dev);
        torch::Tensor token_ids;
        {
            torch::NoGradGuard g;
            token_ids = ch5::generate(model, encoded, /*max_new_tokens=*/100,
                                      /*context_size=*/1024, /*temperature=*/0.0,
                                      /*top_k=*/c10::nullopt, /*eos_id=*/50256);
        }
        auto full = ch5::token_ids_to_text(token_ids.to(torch::kCPU), tokenizer);
        std::cout << full.substr(start_context.size()) << "\n";
        model->train();
    }
}

// ---- 7.8 Ollama 查询 ----
std::string query_ollama(const std::string& prompt, const std::string& model = "llama3",
                         const std::string& url = "http://localhost:11434/api/chat") {
    nlohmann::json data = {
        {"model", model},
        {"messages", nlohmann::json::array({{
            {"role", "user"}, {"content", prompt}
        }})},
        {"options", {{"seed", 123}, {"temperature", 0}, {"num_ctx", 2048}}},
        {"stream", false}};
    std::string payload = data.dump();

    std::string cmd = "curl --noproxy '*' -s -X POST -H 'Content-Type: application/json' -d '" +
                      payload + "' " + url;
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);

    try {
        auto j = nlohmann::json::parse(out);
        return j["message"]["content"].get<std::string>();
    } catch (...) {
        return "";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string data_dir = DATA_DIR;
    if (argc > 1) data_dir = argv[1];
    int64_t num_epochs = 2;
    if (argc > 2) num_epochs = std::stoll(argv[2]);

    try {
        at::globalContext().setAllowTF32CuBLAS(false);
        torch::Device device =
            torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                        : torch::Device(torch::kCPU);
        ch2::BpeTokenizer tokenizer(TOKENIZER_DIR "/encoder.json",
                                    TOKENIZER_DIR "/vocab.bpe");
        std::cout << "=== 第 7 章：通过微调遵循人类指令（C++ + LibTorch）===\n"
                  << "数据目录: " << data_dir << "\n"
                  << "设备: " << device << "\n";

        // ---- 7.2 数据准备 ----
        section("7.2 为有监督指令微调准备数据集");
        auto data = read_instruction_json(data_dir + "/instruction-data.json");
        std::cout << "Number of entries: " << data.size() << "\n";
        std::cout << "Example entry:\n  instruction: " << data[50].instruction
                  << "\n  input: " << data[50].input
                  << "\n  output: " << data[50].output << "\n";
        std::cout << "Another example entry (empty input):\n  instruction: "
                  << data[999].instruction << "\n  output: " << data[999].output << "\n";

        std::cout << "\n-- format_input(data[50]) --\n"
                  << ch7::format_input(data[50]) + "\n\n### Response:\n" + data[50].output
                  << "\n";
        std::cout << "\n-- format_input(data[999])（无 ### Input:）--\n"
                  << ch7::format_input(data[999]) + "\n\n### Response:\n" + data[999].output
                  << "\n";

        // 划分 85/10/5
        size_t train_portion = static_cast<size_t>(data.size() * 0.85);
        size_t test_portion = static_cast<size_t>(data.size() * 0.1);
        std::vector<ch7::InstructionEntry> train_data(data.begin(),
                                                      data.begin() + train_portion);
        std::vector<ch7::InstructionEntry> test_data(data.begin() + train_portion,
                                                     data.begin() + train_portion + test_portion);
        std::vector<ch7::InstructionEntry> val_data(data.begin() + train_portion + test_portion,
                                                    data.end());
        std::cout << "\nTraining set length: " << train_data.size()
                  << ", Validation set length: " << val_data.size()
                  << ", Test set length: " << test_data.size() << "\n";

        // ---- 7.3 批处理 ----
        section("7.3 将数据组织成训练批次");
        {
            // custom_collate_fn 独立测试（与书 7.3 节示例一致）
            std::vector<std::vector<int>> batch = {
                {0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9}};
            auto [in, tgt] = ch7::custom_collate_fn(batch);
            std::cout << "custom_collate_fn(batch) inputs:\n" << in << "\n";
            std::cout << "targets:\n" << tgt << "\n";
        }
        std::cout << "tokenizer.encode(\"<|endoftext|>\", allowed_special) = ";
        auto eot = tokenizer.encode("<|endoftext|>", {"<|endoftext|>"});
        for (int id : eot) std::cout << id;
        std::cout << "（填充词元 ID）\n";

        demo_cross_entropy();

        // ---- 7.4 数据加载器 ----
        section("7.4 创建指令数据集的数据加载器");
        ch7::InstructionDataset train_ds(train_data, tokenizer);
        ch7::InstructionDataset val_ds(val_data, tokenizer);
        ch7::InstructionDataset test_ds(test_data, tokenizer);
        const int64_t batch_size = 8;
        ch7::InstructionLoader train_loader(train_ds, batch_size, /*shuffle=*/true,
                                            /*drop_last=*/true, 50256, 1024, device, 123);
        ch7::InstructionLoader val_loader(val_ds, batch_size, /*shuffle=*/false,
                                          /*drop_last=*/false, 50256, 1024, device, 123);
        ch7::InstructionLoader test_loader(test_ds, batch_size, /*shuffle=*/false,
                                           /*drop_last=*/false, 50256, 1024, device, 123);
        std::cout << "Train loader:\n";
        int count = 0;
        for (const auto& b : train_loader.batches()) {
            std::cout << "  " << b.inputs.sizes() << " " << b.targets.sizes() << "\n";
            if (++count >= 3) { std::cout << "  ...\n"; break; }
        }
        std::cout << "总批次: " << train_loader.num_batches() << " train, "
                  << val_loader.num_batches() << " val, "
                  << test_loader.num_batches() << " test\n";

        // ---- 7.5 加载预训练模型（gpt2-medium 355M）----
        section("7.5 加载预训练的大语言模型（GPT-2 medium 355M）");
        ch4::GPTConfig cfg;
        cfg.context_length = 1024;
        cfg.drop_rate = 0.0;
        cfg.qkv_bias = true;
        cfg.emb_dim = 1024;
        cfg.n_layers = 24;
        cfg.n_heads = 16;
        ch4::GPTModel model(cfg);
        load_gpt2_weights(model, data_dir + "/gpt2-medium.safetensors");
        model->to(device);
        model->eval();
        std::cout << "GPT-2 medium 权重加载完成（设备 " << device << "）\n";

        // 微调前基准：val_data[0]
        std::string input_text = ch7::format_input(val_data[0]);
        std::cout << "\ninput_text (val_data[0]):\n" << input_text << "\n";
        torch::manual_seed(123);
        auto token_ids = ch5::generate(model, ch5::text_to_token_ids(input_text, tokenizer).to(device),
                                       /*max_new_tokens=*/35, /*context_size=*/1024,
                                       /*temperature=*/0.0, c10::nullopt, /*eos_id=*/50256);
        auto generated_text = ch5::token_ids_to_text(token_ids.to(torch::kCPU), tokenizer);
        std::cout << "\n微调前模型回复:\n" << ch7::extract_response(generated_text, input_text)
                  << "\n";

        // ---- 7.6 微调 ----
        section("7.6 在指令数据上微调大语言模型");
        {
            torch::NoGradGuard g;
            double train_loss = calc_loss_loader(model, train_loader.batches(), 5);
            double val_loss = calc_loss_loader(model, val_loader.batches(), 5);
            std::cout << "Training loss: " << train_loss << "\n";
            std::cout << "Validation loss: " << val_loss << "\n";
        }
        torch::manual_seed(123);
        torch::optim::AdamWOptions options(5e-5);
        options.weight_decay(0.1);
        auto optimizer = torch::optim::AdamW(model->parameters(), options);

        std::string start_context = ch7::format_input(val_data[0]);
        train_model_simple(model, train_loader.batches(), val_loader.batches(), optimizer,
                           num_epochs, /*eval_freq=*/5, /*eval_iter=*/5,
                           start_context, tokenizer);

        // ---- 7.7 抽取并保存测试集回复 ----
        section("7.7 抽取并保存模型回复");
        model->eval();
        torch::manual_seed(123);
        for (size_t i = 0; i < test_data.size(); ++i) {
            std::string in_text = ch7::format_input(test_data[i]);
            auto tids = ch5::generate(model, ch5::text_to_token_ids(in_text, tokenizer).to(device),
                                      /*max_new_tokens=*/256, /*context_size=*/1024,
                                      /*temperature=*/0.0, c10::nullopt, /*eos_id=*/50256);
            auto full = ch5::token_ids_to_text(tids.to(torch::kCPU), tokenizer);
            test_data[i].model_response = ch7::extract_response(full, in_text);
            if (i % 20 == 0)
                std::cout << "已生成 " << (i + 1) << "/" << test_data.size() << " 条回复\n";
        }
        save_response_json(data_dir + "/instruction-data-with-response.json", test_data);
        std::cout << "已保存 instruction-data-with-response.json（" << test_data.size()
                  << " 条）\n";

        // 打印前 3 个测试样本对比
        std::cout << "\n-- 测试集前 3 个样本对比 --\n";
        for (size_t i = 0; i < 3 && i < test_data.size(); ++i) {
            std::cout << "\n" << ch7::format_input(test_data[i]) << "\n";
            std::cout << "\nCorrect response:\n>> " << test_data[i].output << "\n";
            std::cout << "\nModel response:\n>> " << test_data[i].model_response << "\n";
            std::cout << "\n-------------------------------------\n";
        }

        // 保存模型
        std::string model_path = data_dir + "/gpt2-medium355M-sft.pth";
        torch::save(model, model_path);
        std::cout << "Model saved as " << model_path << "\n";

        // ---- 7.8 Ollama 评估 ----
        section("7.8 评估微调后的大语言模型");
        {
            std::string check = query_ollama("Reply with 'ok'.");
            if (check.empty()) {
                std::cout << "Ollama 未运行或无法访问（需要启动 `ollama serve` 并拉取 llama3），"
                          << "跳过自动评分。\n"
                          << "可先运行：ollama pull llama3\n";
            } else {
                std::cout << "Ollama running: true\n";
                // 前 3 个样本打分
                for (size_t i = 0; i < 3 && i < test_data.size(); ++i) {
                    const auto& e = test_data[i];
                    std::string prompt =
                        "Given the input `" + ch7::format_input(e) + "` and correct output `" +
                        e.output + "`, score the model response `" + e.model_response +
                        "` on a scale from 0 to 100, where 100 is the best score. "
                        "Respond with the integer number only.";
                    std::string score = query_ollama(prompt);
                    std::cout << "\nDataset response:\n>> " << e.output
                              << "\nModel response:\n>> " << e.model_response
                              << "\nScore:\n>> " << score << "\n\n------------------\n";
                }
            }
        }

        std::cout << "\nCUDA available: " << (torch::cuda::is_available() ? "yes" : "no")
                  << "\n";
        std::cout << "\n=== 第 7 章演示完成 ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
