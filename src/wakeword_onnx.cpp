#include "wakeword_checkpoint.hpp"
#include <ATen/ATen.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace ezlocalai {
namespace {
// ONNX's stable protobuf wire schema. Only model/graph/tensor/value/node and
// integer/string attributes needed by these two architectures are emitted.
// No Python exporter or protobuf runtime is required by the worker.
struct Proto {
    std::string bytes;
    void varint(uint64_t value) { while (value >= 128) { bytes += char((value & 127) | 128); value >>= 7; } bytes += char(value); }
    void integer(int field, int64_t value) { varint(uint64_t(field) << 3); varint(uint64_t(value)); }
    void string(int field, const std::string &value) { varint((uint64_t(field) << 3) | 2); varint(value.size()); bytes += value; }
    void message(int field, const Proto &value) { string(field, value.bytes); }
};
Proto integer_attribute(const std::string &name, int64_t value) {
    Proto attr; attr.string(1, name); attr.integer(20, 2); attr.integer(3, value); return attr;
}
Proto integers_attribute(const std::string &name, std::initializer_list<int64_t> values) {
    Proto attr; attr.string(1, name); attr.integer(20, 7); for (auto value : values) attr.integer(8, value); return attr;
}
Proto string_attribute(const std::string &name, const std::string &value) {
    Proto attr; attr.string(1, name); attr.integer(20, 3); attr.string(4, value); return attr;
}
Proto value_info(const std::string &name, const std::vector<int64_t> &dimensions) {
    Proto value, type, tensor, shape; value.string(1, name); tensor.integer(1, 1);
    for (auto size : dimensions) { Proto dim; if (size < 0) dim.string(2, "batch_size"); else dim.integer(1, size); shape.message(1, dim); }
    tensor.message(2, shape); type.message(1, tensor); value.message(2, type); return value;
}
class Graph {
    Proto graph_;
    std::ofstream data_;
    uint64_t offset_ = 0;
public:
    explicit Graph(const std::filesystem::path &root) : data_(root / "model.onnx.data", std::ios::binary | std::ios::trunc) {
        if (!data_) throw std::runtime_error("Cannot write ONNX tensor data");
        graph_.string(2, "wakeword");
    }
    std::string tensor(const std::string &name, const at::Tensor &input) {
        const auto tensor = input.detach().to(at::kCPU).contiguous(); Proto record;
        for (auto dimension : tensor.sizes()) record.integer(1, dimension);
        const auto dtype = tensor.scalar_type();
        if (dtype != at::kFloat && dtype != at::kLong && dtype != at::kBool) throw std::runtime_error("Unsupported ONNX initializer type");
        record.integer(2, dtype == at::kFloat ? 1 : dtype == at::kLong ? 7 : 9); record.string(8, name);
        std::string bytes(static_cast<const char *>(tensor.const_data_ptr()), tensor.nbytes());
        const uint16_t endian_probe = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1) {
            const size_t width = tensor.element_size();
            for (size_t i = 0; i < bytes.size(); i += width) std::reverse(bytes.begin() + i, bytes.begin() + i + width);
        }
        // Torch's current exporter externalizes initializers of at least 1 KiB.
        if (bytes.size() >= 1024) {
            const auto pair = [&](const std::string &key, const std::string &value) { Proto entry; entry.string(1, key); entry.string(2, value); record.message(13, entry); };
            pair("location", "model.onnx.data"); pair("offset", std::to_string(offset_)); pair("length", std::to_string(bytes.size())); record.integer(14, 1);
            data_.write(bytes.data(), bytes.size()); offset_ += bytes.size();
            if (!data_) throw std::runtime_error("Cannot write ONNX tensor data");
        } else record.string(9, bytes);
        graph_.message(5, record); return name;
    }
    std::string integers(const std::string &name, std::initializer_list<int64_t> values, bool scalar = false) {
        auto value = at::tensor(std::vector<int64_t>(values), at::TensorOptions().dtype(at::kLong));
        if (scalar) value = value.reshape({});
        return tensor(name, value);
    }
    std::string node(const std::string &op, const std::vector<std::string> &inputs, const std::string &output, const std::vector<Proto> &attributes = {}) {
        Proto node; for (const auto &input : inputs) node.string(1, input); node.string(2, output); node.string(3, output); node.string(4, op);
        for (const auto &attr : attributes) node.message(5, attr);
        graph_.message(1, node); return output;
    }
    void save(const std::filesystem::path &root, int64_t n_mfcc) {
        graph_.message(11, value_info("input", {-1, n_mfcc, 150}));
        // Torch's exported output annotation stays at its example batch size,
        // including GRU graphs whose runtime output can have a larger batch.
        graph_.message(12, value_info("output", {1}));
        Proto model, opset; model.integer(1, 10); model.string(2, "ezlocalai-native"); model.message(7, graph_);
        opset.string(1, ""); opset.integer(2, 18); model.message(8, opset);
        data_.close(); if (!data_) throw std::runtime_error("Cannot finish ONNX tensor data");
        std::ofstream file(root / "model.onnx", std::ios::binary); file.write(model.bytes.data(), model.bytes.size());
        if (!file) throw std::runtime_error("Cannot write ONNX model");
    }
};
}
void export_wakeword_onnx(const std::string &directory, torch::nn::Module &module, const nlohmann::json &config) {
    at::NoGradGuard guard;
    const auto root = std::filesystem::path(directory); std::filesystem::create_directories(root); Graph graph(root);
    std::map<std::string, at::Tensor> state;
    for (const auto &item : module.named_parameters()) state[item.key()] = item.value().detach().to(at::kCPU);
    for (const auto &item : module.named_buffers()) state[item.key()] = item.value().detach().to(at::kCPU);
    const auto n_mfcc = config.at("n_mfcc").get<int64_t>(), hidden = config.at("hidden_size").get<int64_t>();
    std::string x;
    const auto dropout = [&](const std::string &input, const std::string &name) {
        if (!module.is_training() || config.at("dropout").get<double>() == 0) return input;
        const auto ratio = graph.tensor(name + ".ratio", at::scalar_tensor(config.at("dropout").get<double>(), at::TensorOptions().dtype(at::kFloat)));
        const auto training = graph.tensor(name + ".training", at::scalar_tensor(true, at::TensorOptions().dtype(at::kBool)));
        return graph.node("Dropout", {input, ratio, training}, name);
    };
    const auto dense = [&](const std::string &input, const std::string &name) {
        const auto weight = graph.tensor(name + ".weight", state.at(name + ".weight"));
        const auto bias = graph.tensor(name + ".bias", state.at(name + ".bias"));
        return graph.node("Gemm", {input, weight, bias}, name, {integer_attribute("transB", 1)});
    };
    if (config.at("model_type") == "gru") {
        x = graph.node("Transpose", {"input"}, "sequence", {integers_attribute("perm", {2, 0, 1})});
        const auto layers = config.at("num_layers").get<int64_t>();
        const auto shape = graph.integers("gru.reshape", {0, 0, -1});
        for (int64_t layer = 0; layer < layers; ++layer) {
            std::vector<at::Tensor> weights, recurrent, biases;
            for (const std::string direction : {"", "_reverse"}) {
                const auto suffix = "_l" + std::to_string(layer) + direction;
                // PyTorch gate order is r,z,n; ONNX requires z,r,h.
                const auto gates = [&](const std::string &name) {
                    const auto value = state.at("gru." + name + suffix);
                    return at::cat({value.slice(0, hidden, 2 * hidden), value.slice(0, 0, hidden), value.slice(0, 2 * hidden)}, 0);
                };
                weights.push_back(gates("weight_ih")); recurrent.push_back(gates("weight_hh"));
                biases.push_back(at::cat({gates("bias_ih"), gates("bias_hh")}, 0));
            }
            const auto name = "gru." + std::to_string(layer);
            const auto weight = graph.tensor(name + ".W", at::stack(weights));
            const auto recurrence = graph.tensor(name + ".R", at::stack(recurrent));
            const auto bias = graph.tensor(name + ".B", at::stack(biases));
            x = graph.node("GRU", {x, weight, recurrence, bias}, name + ".Y", {integer_attribute("hidden_size", hidden), string_attribute("direction", "bidirectional"), integer_attribute("linear_before_reset", 1)});
            x = graph.node("Transpose", {x}, name + ".batch", {integers_attribute("perm", {0, 2, 1, 3})});
            x = graph.node("Reshape", {x, shape}, name + ".sequence");
            if (layer + 1 < layers) x = dropout(x, name + ".dropout");
        }
        x = graph.node("Gather", {x, graph.integers("last_time", {-1}, true)}, "last", {integer_attribute("axis", 0)});
        x = dense(x, "fc");
    } else {
        x = graph.node("Unsqueeze", {"input", graph.integers("channel_axis", {1})}, "channels");
        for (int i = 1; i <= 3; ++i) {
            const auto conv = "conv" + std::to_string(i), bn = "bn" + std::to_string(i);
            // Match Torch/ONNX's evaluation-time Conv+BatchNorm folding. The
            // division by sqrt matters: multiplication by rsqrt differs by ULPs.
            const auto scale = state.at(bn + ".weight") / at::sqrt(state.at(bn + ".running_var") + 1e-5);
            const auto weight = state.at(conv + ".weight") * scale.reshape({-1, 1, 1, 1});
            const auto bias = (state.at(conv + ".bias") - state.at(bn + ".running_mean")) * scale + state.at(bn + ".bias");
            const auto w = graph.tensor(conv + ".weight", weight), b = graph.tensor(conv + ".bias", bias);
            x = graph.node("Conv", {x, w, b}, conv, {integers_attribute("pads", {1, 1, 1, 1}), integers_attribute("kernel_shape", {3, 3})});
            x = graph.node("Relu", {x}, conv + ".relu");
            x = graph.node("MaxPool", {x}, conv + ".pool", {integers_attribute("kernel_shape", {2, 2}), integers_attribute("strides", {2, 2})});
        }
        // The installed Python exporter specializes flatten to its batch-one
        // example despite the dynamic_axes request. Preserve that restriction.
        x = graph.node("Reshape", {x, graph.integers("flat_shape", {1, 64 * (n_mfcc / 8) * (150 / 8)})}, "flat", {integer_attribute("allowzero", 1)});
        x = dense(x, "fc1"); x = graph.node("Relu", {x}, "fc1.relu"); x = dropout(x, "dropout"); x = dense(x, "fc2");
    }
    x = graph.node("Sigmoid", {x}, "probability");
    graph.node("Squeeze", {x, graph.integers("output_axis", {1})}, "output"); graph.save(root, n_mfcc);
}
}
