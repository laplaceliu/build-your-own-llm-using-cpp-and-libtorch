// torchtext_gpt2_bpe.h
// 使用 torchtext 官方 C++ GPT2BPEEncoder 的封装工厂
//
// torchtext（PyTorch 文本库）提供了 GPT-2 BPE 分词器的 C++ 实现
// （torchtext::GPT2BPEEncoder，见 third_party/torchtext/），它重实现了
// openai GPT-2 的 BPE 算法，等价于 Python 的
// torchtext.transforms.GPT2BPETokenizer / tiktoken.get_encoding("gpt2")。
//
// 本文件提供从数据文件构建该分词器的工厂函数：
//   encoder.json —— token -> id
//   vocab.bpe    —— merge 规则（每行 "a b"），行号即 rank
#pragma once

#include <torch/script.h>

#include <memory>
#include <string>

#include "gpt2_bpe_tokenizer.h"

namespace ch2 {

// 从 encoder.json / vocab.bpe 构建 torchtext 官方 GPT2BPEEncoder。
// 返回 c10::intrusive_ptr，可直接调用 Encode / Decode / Tokenize。
c10::intrusive_ptr<torchtext::GPT2BPEEncoder> load_gpt2_bpe(
    const std::string& encoder_json_path,
    const std::string& vocab_bpe_path,
    bool caching_enabled = true);

}  // namespace ch2
