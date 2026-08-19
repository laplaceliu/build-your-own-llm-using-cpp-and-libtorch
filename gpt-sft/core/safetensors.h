// safetensors.h
// 轻量级 safetensors 文件解析器（用于 5.5 节加载 HuggingFace gpt2 权重）
//
// safetensors 二进制格式：
//   [8 字节 uint64 LE] header 长度 N
//   [N 字节]            header JSON（{"name": {"dtype","shape","data_offsets"}}）
//   [数据区]            各张量按 data_offsets 连续存放（float32, row-major）
#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>

namespace ch5 {

// 解析 safetensors 文件，返回 张量名 -> Tensor（float32, CPU）
std::unordered_map<std::string, torch::Tensor> load_safetensors(
    const std::string& path);

}  // namespace ch5
