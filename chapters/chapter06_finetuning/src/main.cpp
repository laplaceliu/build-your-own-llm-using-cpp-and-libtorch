// main.cpp
// 第 6 章：针对分类的微调（C++ + LibTorch 实现）
//
// 按书中步骤演示，中间结果全部打印：
//   6.2  下载数据集 / 平衡 / 划分（train.csv, validation.csv, test.csv）
//   6.3  创建数据加载器（SpamDataset，max_length=120，130/19/38 批次）
//   6.4  加载 GPT-2 预训练权重（safetensors）并验证文本生成
//   6.5  添加分类头（替换 out_head + 冻结/解冻参数）
//   6.6  计算分类损失和准确率（初始 ~46%/45%/49%，损失 2.453/2.583/2.322）
//   6.7  在有监督数据上微调（AdamW 5 轮）
//   6.8  使用微调模型分类新数据 + 保存模型
//
// 用法：
//   ./chapter06_finetuning [data_dir] [safetensors]
//     data_dir     默认 data/（含 SMSSpamCollection.tsv）
//     safetensors  默认 chapter02 的 gpt2-model.hf.safetensors
#include <torch/torch.h>

#include <iostream>
#include <string>

#include "bpe_tokenizer.h"
#include "finetuning.h"
#include "safetensors.h"
#include "training.h"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif
#ifndef TOKENIZER_DIR
#define TOKENIZER_DIR "../chapter02_text_data/data"
#endif
#ifndef GPT2_WEIGHTS
#define GPT2_WEIGHTS TOKENIZER_DIR "/gpt2-model.hf.safetensors"
#endif

namespace {

void section(const std::string& title) {
    std::cout << "\n========================================================\n"
              << title << "\n"
              << "========================================================\n";
}

// 选择计算设备：优先 CUDA（RTX 4080），否则 CPU
torch::Device select_device() {
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
}

// 把一批 (inputs, labels) 迁移到指定设备
std::vector<ch6::SpamBatch> to_device(const std::vector<ch6::SpamBatch>& loader,
                                      const torch::Device& dev) {
    std::vector<ch6::SpamBatch> out;
    out.reserve(loader.size());
    for (const auto& b : loader) {
        out.push_back({b.inputs.to(dev), b.labels.to(dev)});
    }
    return out;
}

// 5.5 节权重加载逻辑（HF safetensors -> GPTModel）
void load_gpt2_weights(ch4::GPTModel& gpt, const std::string& st_path) {
    auto st = ch5::load_safetensors(st_path);

    // 书中 params 键名 -> HF safetensors 键名
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
            throw std::runtime_error("Shape mismatch");
        {
            torch::NoGradGuard g;
            left.copy_(right);
        }
        return left;
    };
    auto params = [&](const std::string& name) -> const torch::Tensor& {
        return st.at(map_key(name));
    };

    gpt->pos_emb->weight = assign(gpt->pos_emb->weight, params("wpe"));
    gpt->tok_emb->weight = assign(gpt->tok_emb->weight, params("wte"));

    int64_t emb = gpt->tok_emb->weight.size(1);
    for (int64_t b = 0; b < static_cast<int64_t>(gpt->trf_blocks->size()); ++b) {
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

}  // namespace

int main(int argc, char* argv[]) {
    std::string data_dir = DATA_DIR;
    if (argc > 1) data_dir = argv[1];
    std::string weights = GPT2_WEIGHTS;
    if (argc > 2) weights = argv[2];

    try {
        ch2::BpeTokenizer tokenizer(TOKENIZER_DIR "/encoder.json",
                                    TOKENIZER_DIR "/vocab.bpe");
        torch::Device device = select_device();
        std::cout << "=== 第 6 章：针对分类的微调（C++ + LibTorch）===\n"
                  << "数据目录: " << data_dir << "\n"
                  << "权重文件: " << weights << "\n"
                  << "设备: " << (device.is_cuda() ? "CUDA (RTX 4080)" : "CPU")
                  << (device.is_cuda() ? "" : (", 线程 " + std::to_string(torch::get_num_threads())))
                  << "\n";

        // 6.2 数据集准备
        section("6.2 准备数据集");
        {
            auto df = ch6::read_spam_tsv(data_dir + "/SMSSpamCollection.tsv");
            std::cout << "总样本数: " << df.size() << "\n";
            int ham = 0, spam = 0;
            for (const auto& it : df) (it.label == 1 ? spam : ham)++;
            std::cout << "标签分布: ham = " << ham << ", spam = " << spam << "\n";
            std::cout << "前 3 条样例:\n";
            for (int i = 0; i < 3; ++i) {
                std::cout << "  [" << (df[i].label == 1 ? "spam" : "ham") << "] "
                          << df[i].text.substr(0, 60) << "...\n";
            }

            auto balanced = ch6::create_balanced_dataset(df);
            ham = spam = 0;
            for (const auto& it : balanced) (it.label == 1 ? spam : ham)++;
            std::cout << "\n平衡后: ham = " << ham << ", spam = " << spam
                      << "（总计 " << balanced.size() << "）\n";

            auto [train, val, test] = ch6::random_split(balanced, 0.7, 0.1);
            std::cout << "划分: train = " << train.size() << ", val = " << val.size()
                      << ", test = " << test.size() << "\n";

            ch6::save_csv(data_dir + "/train.csv", train);
            ch6::save_csv(data_dir + "/validation.csv", val);
            ch6::save_csv(data_dir + "/test.csv", test);
            std::cout << "已保存 train.csv / validation.csv / test.csv\n";
            auto first3 = ch6::read_csv(data_dir + "/train.csv");
            std::cout << "train.csv 前 3 行:\n";
            for (int i = 0; i < 3; ++i) {
                std::cout << "  [" << (first3[i].label == 1 ? "spam" : "ham") << "] "
                          << first3[i].text.substr(0, 60) << "...\n";
            }
        }

        // 6.3 数据加载器
        section("6.3 创建数据加载器");
        ch6::SpamDataset train_dataset(data_dir + "/train.csv", tokenizer);
        std::cout << "训练集 max_length = " << train_dataset.max_length() << "\n";
        ch6::SpamDataset val_dataset(data_dir + "/validation.csv", tokenizer,
                                     train_dataset.max_length());
        ch6::SpamDataset test_dataset(data_dir + "/test.csv", tokenizer,
                                      train_dataset.max_length());
        const int64_t batch_size = 8;
        auto train_loader = ch6::make_spam_loader(train_dataset, batch_size,
                                                  /*shuffle=*/true, /*drop_last=*/true, 123);
        auto val_loader = ch6::make_spam_loader(val_dataset, batch_size,
                                                /*shuffle=*/false, /*drop_last=*/false, 123);
        auto test_loader = ch6::make_spam_loader(test_dataset, batch_size,
                                                 /*shuffle=*/false, /*drop_last=*/false, 123);
        std::cout << "最后一个训练批次: inputs " << train_loader.back().inputs.sizes()
                  << ", labels " << train_loader.back().labels.sizes() << "\n";
        std::cout << "批次数量: " << train_loader.size() << " training, "
                  << val_loader.size() << " validation, " << test_loader.size()
                  << " test\n";

        // 批次迁移到计算设备
        auto train_loader_dev = to_device(train_loader, device);
        auto val_loader_dev = to_device(val_loader, device);
        auto test_loader_dev = to_device(test_loader, device);

        // 6.4 预训练模型
        section("6.4 初始化带有预训练权重的模型");
        ch4::GPTConfig cfg;
        cfg.context_length = 1024;
        cfg.drop_rate = 0.0;
        cfg.qkv_bias = true;
        ch4::GPTModel model(cfg);
        load_gpt2_weights(model, weights);
        model->to(device);  // 迁移到 GPU
        model->eval();
        std::cout << "GPT-2 small 权重加载完成（设备: "
                  << (device.is_cuda() ? "cuda:0" : "cpu") << "）\n";

        auto text_1 = "Every effort moves you";
        auto token_ids = ch4::generate_text_simple(model,
                                                   ch5::text_to_token_ids(text_1, tokenizer).to(device),
                                                   /*max_new_tokens=*/15,
                                                   /*context_size=*/1024);
        std::cout << "生成文本: " << ch5::token_ids_to_text(token_ids.to(torch::kCPU), tokenizer) << "\n";
        auto text_2 = std::string(
            "Is the following text 'spam'? Answer with 'yes' or 'no': "
            "'You are a winner you have been specially selected to receive "
            "$1000 cash or a $2000 award.'");
        token_ids = ch4::generate_text_simple(model,
                                              ch5::text_to_token_ids(text_2, tokenizer).to(device),
                                              /*max_new_tokens=*/23,
                                              /*context_size=*/1024);
        std::cout << "指令输入生成（未微调，不遵循指令）:\n"
                  << ch5::token_ids_to_text(token_ids.to(torch::kCPU), tokenizer) << "\n";

        // 6.5 分类头
        section("6.5 添加分类头");
        torch::manual_seed(123);
        ch6::GPTClassifier classifier(model, /*num_classes=*/2);
        classifier->to(device);  // 新输出层也迁移到 GPU
        std::cout << "out_head: Linear(" << classifier->out_head_->weight.size(1)
                  << " -> 2)\n";
        std::cout << "可训练参数数量: " << classifier->trainable_parameters().size()
                  << "（其余已冻结）\n";
        auto inputs = tokenizer.encode("Do you have time");
        auto input_tensor = torch::tensor(inputs, torch::kLong).to(device).unsqueeze(0);
        std::cout << "Inputs: " << input_tensor << "\n";
        std::cout << "Inputs dimensions: " << input_tensor.sizes() << "\n";
        torch::Tensor outputs;
        {
            torch::NoGradGuard g;
            outputs = classifier->forward(input_tensor);
        }
        std::cout << "Outputs:\n" << outputs.to(torch::kCPU) << "\n";
        std::cout << "Outputs dimensions: " << outputs.sizes() << "\n";
        std::cout << "Last output token: "
                  << outputs.index({torch::indexing::Slice(), -1, torch::indexing::Slice()})
                         .to(torch::kCPU)
                  << "\n";

        // 6.6 初始评估
        section("6.6 计算分类损失和准确率");
        double train_acc = ch6::calc_accuracy_loader(classifier, train_loader_dev, 10);
        double val_acc = ch6::calc_accuracy_loader(classifier, val_loader_dev, 10);
        double test_acc = ch6::calc_accuracy_loader(classifier, test_loader_dev, 10);
        std::cout << "Training accuracy: " << (train_acc * 100.0) << "%\n";
        std::cout << "Validation accuracy: " << (val_acc * 100.0) << "%\n";
        std::cout << "Test accuracy: " << (test_acc * 100.0) << "%\n";
        double train_loss, val_loss, test_loss;
        {
            torch::NoGradGuard g;
            train_loss = ch6::calc_loss_loader(classifier, train_loader_dev, 5);
            val_loss = ch6::calc_loss_loader(classifier, val_loader_dev, 5);
            test_loss = ch6::calc_loss_loader(classifier, test_loader_dev, 5);
        }
        std::cout << "Training loss: " << train_loss << "\n";
        std::cout << "Validation loss: " << val_loss << "\n";
        std::cout << "Test loss: " << test_loss << "\n";

        // 6.7 微调
        section("6.7 在有监督数据上微调模型");
        torch::manual_seed(123);
        torch::optim::AdamWOptions options(5e-5);
        options.weight_decay(0.1);
        auto optimizer = torch::optim::AdamW(classifier->trainable_parameters(), options);
        std::cout << "开始微调 5 轮（设备: "
                  << (device.is_cuda() ? "cuda:0" : "cpu") << "）...\n";
        ch6::train_classifier_simple(classifier, train_loader_dev, val_loader_dev, optimizer,
                                     /*num_epochs=*/5, /*eval_freq=*/50, /*eval_iter=*/5);

        // 6.8 分类新数据
        section("6.8 使用大语言模型作为垃圾消息分类器");
        const std::string cl_text_1 =
            "You are a winner you have been specially selected to receive "
            "$1000 cash or a $2000 award.";
        std::cout << "text_1: " << cl_text_1 << "\n";
        std::cout << "分类结果: "
                  << ch6::classify_review(cl_text_1, classifier, tokenizer,
                                          train_dataset.max_length())
                  << "\n";
        const std::string cl_text_2 =
            "Hey, just wanted to check if we're still on for dinner tonight? Let me know!";
        std::cout << "text_2: " << cl_text_2 << "\n";
        std::cout << "分类结果: "
                  << ch6::classify_review(cl_text_2, classifier, tokenizer,
                                          train_dataset.max_length())
                  << "\n";

        std::string model_path = data_dir + "/review_classifier.pth";
        torch::save(classifier, model_path);
        std::cout << "已保存分类模型到 " << model_path << "\n";

        std::cout << "\nCUDA available: " << (torch::cuda::is_available() ? "yes" : "no")
                  << "\n";
        std::cout << "\n=== 第 6 章演示完成 ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
