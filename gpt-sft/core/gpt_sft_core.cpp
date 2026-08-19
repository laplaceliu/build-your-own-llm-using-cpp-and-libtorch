// gpt_sft_core.cpp
#include "gpt_sft_core.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include <nlohmann/json.hpp>

#include "training.h"  // ch5::generate / text_to_token_ids / token_ids_to_text

namespace gpt_sft {

ModelConfig config_for_size(const std::string& size) {
  ModelConfig cfg;
  if (size == "small") {
    cfg.emb_dim = 768; cfg.n_heads = 12; cfg.n_layers = 12;
  } else if (size == "large") {
    cfg.emb_dim = 1280; cfg.n_heads = 20; cfg.n_layers = 36;
  } else if (size == "xl") {
    cfg.emb_dim = 1600; cfg.n_heads = 25; cfg.n_layers = 48;
  } else {  // default medium
    cfg.emb_dim = 1024; cfg.n_heads = 16; cfg.n_layers = 24;
  }
  return cfg;
}

ch4::GPTModel make_model(const ModelConfig& cfg) {
  ch4::GPTConfig g;
  g.vocab_size = cfg.vocab_size;
  g.context_length = cfg.context_length;
  g.emb_dim = cfg.emb_dim;
  g.n_heads = cfg.n_heads;
  g.n_layers = cfg.n_layers;
  g.drop_rate = cfg.drop_rate;
  g.qkv_bias = cfg.qkv_bias;
  return ch4::GPTModel(g);
}

torch::Device select_device() {
  if (torch::cuda::is_available()) return torch::Device(torch::kCUDA, 0);
  return torch::Device(torch::kCPU);
}

// ---------------------------------------------------------------------------
// HF safetensors -> GPTModel 权重加载（与第 5/7 章一致）
// ---------------------------------------------------------------------------
namespace {
// 书中 params 键名 -> HF safetensors 键名
std::string map_key(const std::string& k) {
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
}

torch::Tensor assign(torch::Tensor left, const torch::Tensor& right) {
  if (left.sizes() != right.sizes())
    throw std::runtime_error("Shape mismatch: " + std::to_string(left.numel()) +
                             " vs " + std::to_string(right.numel()));
  { torch::NoGradGuard g; left.copy_(right); }
  return left;
}
}  // namespace

ch4::GPTModel load_pretrained(const ModelConfig& cfg, const std::string& safetensors_path) {
  auto st = ch5::load_safetensors(safetensors_path);
  auto gpt = make_model(cfg);
  int64_t n_layers = gpt->trf_blocks->size();
  int64_t emb = gpt->tok_emb->weight.size(1);

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
  gpt->out_head->weight = assign(gpt->out_head->weight, params("wte"));  // weight tying
  return gpt;
}

// ---------------------------------------------------------------------------
// 指令数据
// ---------------------------------------------------------------------------
std::vector<ch7::InstructionEntry> load_instruction_data(const std::string& path) {
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

void split_dataset(const std::vector<ch7::InstructionEntry>& data,
                   std::vector<ch7::InstructionEntry>& train,
                   std::vector<ch7::InstructionEntry>& val,
                   std::vector<ch7::InstructionEntry>& test) {
  size_t train_portion = static_cast<size_t>(data.size() * 0.85);
  size_t test_portion = static_cast<size_t>(data.size() * 0.1);
  train.assign(data.begin(), data.begin() + train_portion);
  test.assign(data.begin() + train_portion, data.begin() + train_portion + test_portion);
  val.assign(data.begin() + train_portion + test_portion, data.end());
}

// ---------------------------------------------------------------------------
// 指令微调训练
// ---------------------------------------------------------------------------
namespace {
torch::Tensor loss_batch(const torch::Tensor& in, const torch::Tensor& tgt,
                         ch4::GPTModel& model) {
  auto logits = model->forward(in);
  return torch::nn::functional::cross_entropy(
      logits.flatten(0, 1), tgt.flatten(),
      torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));
}
}  // namespace

double train_instruction(ch4::GPTModel& model,
                         const std::vector<ch7::InstructionEntry>& train_data,
                         const std::vector<ch7::InstructionEntry>& val_data,
                         ch2::BpeTokenizer& tokenizer,
                         int64_t num_epochs, int64_t batch_size, double lr,
                         int64_t eval_freq, int64_t eval_iter,
                         const std::string& start_context,
                         const torch::Device& device) {
  ch7::InstructionDataset train_ds(train_data, tokenizer);
  ch7::InstructionDataset val_ds(val_data, tokenizer);
  ch7::InstructionLoader train_loader(train_ds, batch_size, /*shuffle=*/true,
                                      /*drop_last=*/true, 50256, 1024, device, 123);
  ch7::InstructionLoader val_loader(val_ds, batch_size, /*shuffle=*/false,
                                    /*drop_last=*/false, 50256, 1024, device, 123);

  // 初始损失
  auto calc = [&](const std::vector<ch7::InstructionBatch>& loader, int64_t n) {
    double total = 0.0;
    torch::NoGradGuard g;
    model->eval();
    n = std::min(n, static_cast<int64_t>(loader.size()));
    for (int64_t i = 0; i < n; ++i)
      total += loss_batch(loader[i].inputs, loader[i].targets, model).item<double>();
    model->train();
    return total / n;
  };
  std::cout << "Training loss: " << calc(train_loader.batches(), eval_iter) << "\n";
  std::cout << "Validation loss: " << calc(val_loader.batches(), eval_iter) << "\n";

  torch::manual_seed(123);
  torch::optim::AdamWOptions options(lr);
  options.weight_decay(0.1);
  auto optimizer = torch::optim::AdamW(model->parameters(), options);

  int64_t global_step = -1;
  for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
    model->train();
    for (const auto& batch : train_loader.batches()) {
      optimizer.zero_grad();
      auto loss = loss_batch(batch.inputs, batch.targets, model);
      loss.backward();
      optimizer.step();
      global_step += 1;
      if (global_step % eval_freq == 0) {
        std::cout << "Ep " << (epoch + 1) << " (Step " << global_step << "): Train loss "
                  << calc(train_loader.batches(), eval_iter) << ", Val loss "
                  << calc(val_loader.batches(), eval_iter) << "\n";
      }
    }
    // 每轮结束生成一个示例（val_data[0]）
    if (!val_data.empty()) {
      auto reply = generate_response(model, tokenizer, val_data[0].instruction,
                                     val_data[0].input, /*max_new_tokens=*/100, device);
      std::cout << "  [示例] " << reply << "\n";
    }
  }
  return calc(train_loader.batches(), eval_iter);
}

// ---------------------------------------------------------------------------
// 推理
// ---------------------------------------------------------------------------
std::string generate_response(ch4::GPTModel& model, ch2::BpeTokenizer& tokenizer,
                              const std::string& instruction, const std::string& input,
                              int64_t max_new_tokens, const torch::Device& device) {
  ch7::InstructionEntry entry{instruction, input, "", ""};
  std::string prompt = ch7::format_input(entry);
  auto encoded = ch5::text_to_token_ids(prompt, tokenizer).to(device);
  torch::Tensor tids;
  {
    torch::NoGradGuard g;
    model->eval();
    tids = ch5::generate(model, encoded, max_new_tokens, model->pos_emb->weight.size(0),
                         /*temperature=*/0.0, c10::nullopt, /*eos_id=*/50256);
    model->train();
  }
  auto full = ch5::token_ids_to_text(tids.to(torch::kCPU), tokenizer);
  return ch7::extract_response(full, prompt);
}

}  // namespace gpt_sft
