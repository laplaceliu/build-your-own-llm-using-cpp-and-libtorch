// safetensors.cpp
#include "safetensors.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace ch5 {

std::unordered_map<std::string, torch::Tensor> load_safetensors(
    const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("无法打开 safetensors 文件: " + path);
    }

    // 1. 读取 8 字节 header 长度
    uint64_t header_len = 0;
    ifs.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!ifs || header_len == 0 || header_len > (1u << 30)) {
        throw std::runtime_error("safetensors 头部长度非法: " + path);
    }

    // 2. 读取 header JSON
    std::string header_json(header_len, '\0');
    ifs.read(header_json.data(), static_cast<std::streamsize>(header_len));
    if (!ifs) {
        throw std::runtime_error("safetensors 头部读取失败: " + path);
    }
    auto header = nlohmann::json::parse(header_json);

    // 3. 读取数据区
    ifs.seekg(0, std::ios::end);
    std::streampos file_end = ifs.tellg();
    size_t data_start = static_cast<size_t>(8 + header_len);
    size_t data_size = static_cast<size_t>(file_end) - data_start;
    ifs.seekg(static_cast<std::streamoff>(data_start), std::ios::beg);

    std::vector<char> data(data_size);
    ifs.read(data.data(), static_cast<std::streamsize>(data_size));

    // 4. 按 header 提取每个张量
    std::unordered_map<std::string, torch::Tensor> tensors;
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const auto& meta = it.value();
        std::string dtype = meta["dtype"].get<std::string>();
        if (dtype != "F32" && dtype != "F16") {
            throw std::runtime_error("不支持的 dtype: " + dtype + " (" + it.key() + ")");
        }
        std::vector<int64_t> shape;
        for (const auto& s : meta["shape"]) shape.push_back(s.get<int64_t>());
        uint64_t a = meta["data_offsets"][0].get<uint64_t>();
        uint64_t b = meta["data_offsets"][1].get<uint64_t>();
        uint64_t nbytes = b - a;

        torch::Tensor t;
        if (dtype == "F32") {
            auto* ptr = reinterpret_cast<const float*>(data.data() + a);
            t = torch::from_blob(const_cast<float*>(ptr), shape,
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone();  // clone：数据生命周期独立于文件缓冲
        } else {  // F16
            auto* ptr = reinterpret_cast<const c10::Half*>(data.data() + a);
            t = torch::from_blob(const_cast<c10::Half*>(ptr), shape,
                                 torch::TensorOptions().dtype(torch::kFloat16))
                    .clone();
        }
        (void)nbytes;
        tensors[it.key()] = t;
    }
    return tensors;
}

}  // namespace ch5
