#include "wakeword_checkpoint.hpp"
#include <caffe2/serialize/inline_container.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ezlocalai {
namespace {
// Protocol 2 descriptors understood by torch.load(weights_only=True). The
// numerical storages are separate ZIP records, as in torch.save's default.
class Pickle {
public:
    std::string bytes{"\x80\x02", 2};
    void little(uint64_t value, size_t count) { for (size_t i = 0; i < count; ++i) bytes += char(value >> (8 * i)); }
    void integer(int64_t value) {
        if (value >= 0 && value <= 255) { bytes += 'K'; little(value, 1); }
        else if (value >= 0 && value <= 65535) { bytes += 'M'; little(value, 2); }
        else if (value >= INT32_MIN && value <= INT32_MAX) { bytes += 'J'; little(uint32_t(value), 4); }
        else { bytes += char(0x8a); bytes += char(8); little(uint64_t(value), 8); }
    }
    void string(const std::string &value) {
        if (value.size() > UINT32_MAX) throw std::overflow_error("Checkpoint string exceeds protocol limit");
        bytes += 'X'; little(value.size(), 4); bytes += value;
    }
    void global(const char *module, const char *name) { bytes += 'c'; bytes += module; bytes += '\n'; bytes += name; bytes += '\n'; }
    void ordered_dict() { global("collections", "OrderedDict"); bytes += ")R"; }
    void dimensions(at::IntArrayRef values) { bytes += '('; for (auto value : values) integer(value); bytes += 't'; }
    void json(const nlohmann::json &value) {
        if (value.is_null()) bytes += 'N';
        else if (value.is_boolean()) bytes += char(value.get<bool>() ? 0x88 : 0x89);
        else if (value.is_number_integer()) integer(value.get<int64_t>());
        else if (value.is_number_float()) {
            const auto number = value.get<double>(); uint64_t bits; std::memcpy(&bits, &number, 8); bytes += 'G';
            for (int i = 7; i >= 0; --i) bytes += char(bits >> (i * 8));
        } else if (value.is_string()) string(value.get<std::string>());
        else if (value.is_array()) { bytes += "]("; for (const auto &item : value) json(item); bytes += 'e'; }
        else if (value.is_object()) { bytes += "}("; for (auto it = value.begin(); it != value.end(); ++it) { string(it.key()); json(it.value()); } bytes += 'u'; }
        else throw std::runtime_error("Unsupported checkpoint configuration value");
    }
    void tensor(const at::Tensor &tensor, const std::string &key, const std::string &device) {
        const char *storage = nullptr;
        if (tensor.scalar_type() == at::kFloat) storage = "FloatStorage";
        if (tensor.scalar_type() == at::kLong) storage = "LongStorage";
        if (!storage) throw std::runtime_error("Unsupported wake-word checkpoint dtype");
        global("torch._utils", "_rebuild_tensor_v2"); bytes += "((";
        string("storage"); global("torch", storage); string(key); string(device); integer(tensor.numel()); bytes += "tQ";
        integer(0); dimensions(tensor.sizes()); dimensions(tensor.strides()); bytes += char(0x89); ordered_dict(); bytes += "tR";
    }
};
}
void save_wakeword_checkpoint(const std::string &directory, torch::nn::Module &module, const nlohmann::json &config) {
    const auto root = std::filesystem::path(directory); std::filesystem::create_directories(root);
    caffe2::serialize::PyTorchStreamWriter archive((root / "model.pt").string());
    Pickle pickle; pickle.bytes += '}'; pickle.string("model_state_dict"); pickle.ordered_dict(); pickle.bytes += '(';
    size_t index = 0;
    const auto store = [&](const std::string &name, const at::Tensor &value) {
        const auto tensor = value.detach().to(at::kCPU).contiguous(); const auto key = std::to_string(index++);
        pickle.string(name); pickle.tensor(tensor, key, value.device().str());
        archive.writeRecord("data/" + key, tensor.const_data_ptr(), tensor.nbytes());
    };
    // state_dict traverses modules in registration order, parameters then
    // buffers within each module (rather than all parameters before buffers).
    for (const auto &item : module.named_modules()) {
        const auto prefix = item.key().empty() ? "" : item.key() + ".";
        for (const auto &parameter : item.value()->named_parameters(false)) store(prefix + parameter.key(), parameter.value());
        for (const auto &buffer : item.value()->named_buffers(false)) store(prefix + buffer.key(), buffer.value());
    }
    pickle.bytes += "us"; pickle.string("config"); pickle.json(config); pickle.bytes += "s.";
    archive.writeRecord("data.pkl", pickle.bytes.data(), pickle.bytes.size());
    const uint16_t probe = 1; const std::string order = *reinterpret_cast<const uint8_t *>(&probe) ? "little" : "big";
    archive.writeRecord("byteorder", order.data(), order.size()); archive.writeEndOfFile();
    std::ofstream output(root / "config.json"); output << config.dump(2); if (!output) throw std::runtime_error("Cannot write wake-word configuration");
}
}
