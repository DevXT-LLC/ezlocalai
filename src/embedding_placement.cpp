#include "ezlocalai/embedding_placement.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ezlocalai_c.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <mutex>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "ezlocalai/llm_memory.hpp"

namespace ezlocalai {
namespace {
constexpr double gib = 1024.0 * 1024.0 * 1024.0;
std::string lower(std::string value) { for (char &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return value; }
bool utf8(const std::string &s) {
    for (size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i++]);
        if (c < 128) continue;
        unsigned n = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : 0;
        if (!n || i + n > s.size()) return false;
        uint32_t point = c & ((1u << (6 - n)) - 1);
        for (unsigned j = 0; j < n; ++j) { const auto b = static_cast<unsigned char>(s[i++]); if ((b & 0xc0) != 0x80) return false; point = (point << 6) | (b & 63); }
        if (point < (n == 1 ? 0x80u : n == 2 ? 0x800u : 0x10000u) || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) return false;
    }
    return true;
}
struct Value {
    long double number = 0;
    bool bytes = false;
    std::string text;
    bool integral = true;
    bool truthy() const { return bytes ? !text.empty() : number != 0; }
    int64_t integer() const {
        if (bytes) { if (text.size() != 1) throw std::runtime_error("Non-scalar GGUF metadata"); return static_cast<unsigned char>(text[0]); }
        if (!std::isfinite(number) || number < std::numeric_limits<int64_t>::min() || number > std::numeric_limits<int64_t>::max()) throw std::runtime_error("GGUF integer outside native range");
        return static_cast<int64_t>(number);
    }
    std::string architecture() const {
        if (bytes && utf8(text)) return text;
        if (bytes && text.size() != 1) {
            std::string result = "[";
            for (const auto byte : text) { if (result.size() > 1) result += ", "; result += std::to_string(static_cast<unsigned char>(byte)); }
            return result + ']';
        }
        if (bytes) return std::to_string(integer());
        if (integral && number >= 0 && number < 128) return std::string(1, static_cast<char>(number));
        char buffer[128];
        if (integral) {
            const auto result = number >= 0
                ? std::to_chars(buffer, buffer + sizeof(buffer), static_cast<uint64_t>(number))
                : std::to_chars(buffer, buffer + sizeof(buffer), static_cast<int64_t>(number));
            return {buffer, result.ptr};
        }
        if (std::isnan(number)) return "nan";
        const double value = static_cast<double>(number), magnitude = std::abs(value);
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
            magnitude >= .0001 && magnitude < 1e16 ? std::chars_format::fixed : std::chars_format::general);
        std::string representation(buffer, result.ptr);
        if (std::isfinite(value) && representation.find_first_of(".e") == std::string::npos) representation += ".0";
        return representation;
    }
};
// Read metadata without mapping model weights into the native server. Tensor
// descriptors and bounds are checked too: GGUFReader fails before returning
// metadata if any tensor is invalid. Arrays deliberately keep their final part,
// matching Embedding._field_value rather than interpreting the entire array.
class Gguf {
    std::ifstream input_;
    uint64_t size_, offset_ = 0;
    bool big_ = false;
    bool complete_ = false;
    nlohmann::json contents_;
    uint64_t remaining() const { return offset_ < size_ ? size_ - offset_ : 0; }
    uint64_t number(unsigned bytes) {
        if (remaining() < bytes) throw std::runtime_error("Truncated GGUF scalar");
        unsigned char data[8]{}; input_.read(reinterpret_cast<char *>(data), bytes); offset_ += bytes;
        if (!input_) throw std::runtime_error("Cannot read GGUF");
        uint64_t value = 0;
        for (unsigned i = 0; i < bytes; ++i) value |= uint64_t(data[i]) << (8 * (big_ ? bytes - i - 1 : i));
        return value;
    }
    std::string string() {
        const auto requested = number(8);
        const auto count = std::min(requested, remaining());
        std::string result(static_cast<size_t>(count), '\0');
        input_.read(result.data(), static_cast<std::streamsize>(count)); offset_ += count;
        if (!input_) throw std::runtime_error("Cannot read GGUF string");
        return result;
    }
    Value value(uint32_t type, unsigned depth = 0) {
        if (depth > 128) throw std::runtime_error("Excessively nested GGUF array");
        if (type == 8) { auto text = string(); if (complete_) { if (!utf8(text)) throw std::runtime_error("Invalid GGUF metadata UTF-8"); contents_ = text; } return {0, true, std::move(text)}; }
        if (type == 9) {
            const auto element = number(4), count = number(8);
            if (count > remaining()) throw std::runtime_error("Truncated GGUF array");
            Value last;
            auto items = nlohmann::json::array();
            for (uint64_t i = 0; i < count; ++i) { last = value(element, depth + 1); if (complete_) items.push_back(contents_); }
            if (complete_) contents_ = std::move(items);
            return last;
        }
        unsigned width;
        switch (type) { case 0: case 1: case 7: width = 1; break; case 2: case 3: width = 2; break; case 4: case 5: case 6: width = 4; break; case 10: case 11: case 12: width = 8; break; default: throw std::runtime_error("Invalid GGUF field type"); }
        const uint64_t bits = number(width);
        if (complete_) contents_ = bits;
        if (type == 6) { const auto raw = static_cast<uint32_t>(bits); float f; std::memcpy(&f, &raw, 4); if (complete_) contents_ = f; return {f, false, {}, false}; }
        if (type == 12) { double f; std::memcpy(&f, &bits, 8); if (complete_) contents_ = f; return {f, false, {}, false}; }
        if (type == 7) { if (complete_) contents_ = bits != 0; return {bits != 0 ? 1.0L : 0.0L, false, {}}; }
        if (type == 1 || type == 3 || type == 5 || type == 11) {
            const uint64_t sign = uint64_t(1) << (width * 8 - 1);
            const long double signed_value = bits & sign ? static_cast<long double>(bits ^ sign) - sign : static_cast<long double>(bits);
            if (complete_) contents_ = static_cast<int64_t>(signed_value);
            return {signed_value, false, {}};
        }
        return {static_cast<long double>(bits), false, {}};
    }
public:
    nlohmann::json metadata = nlohmann::json::object();
    std::vector<std::string> tensor_names;
    explicit Gguf(const std::string &path, bool complete = false) : input_(path, std::ios::binary), size_(std::filesystem::file_size(path)), complete_(complete) {}
    std::map<std::string, Value> fields() {
        if (number(4) != 0x46554747) throw std::runtime_error("Invalid GGUF magic");
        auto version = number(4);
        if (!(version & 65535)) { big_ = true; version >>= 24; }
        if (version != 2 && version != 3) throw std::runtime_error("Unsupported GGUF version");
        const auto tensors = number(8), count = number(8);
        if (count > remaining() / 12 || tensors > remaining() / 24) throw std::runtime_error("Truncated GGUF header");
        std::map<std::string, Value> result;
        std::set<std::string> keys{"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"};
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < count; ++i) {
            auto key = string(); const auto type = number(4);
            if (!utf8(key) || !keys.insert(key).second) throw std::runtime_error("Invalid GGUF key");
            auto entry = value(type);
            if (complete_) metadata[key] = contents_;
            if (key == "general.alignment") {
                if (type != 4) throw std::runtime_error("Invalid GGUF alignment type");
                alignment = entry.integer();
                if (!alignment || (alignment & (alignment - 1))) throw std::runtime_error("Invalid GGUF alignment");
            }
            result.emplace(std::move(key), std::move(entry));
        }
        struct Tensor { uint64_t offset, bytes; };
        std::vector<Tensor> data;
        std::set<std::string> names;
        // GGML block and storage sizes from gguf 0.19.0, the reference oracle.
        static const std::map<uint64_t, std::pair<uint64_t, uint64_t>> sizes{
            {0,{1,4}},{1,{1,2}},{2,{32,18}},{3,{32,20}},{6,{32,22}},{7,{32,24}},{8,{32,34}},{9,{32,40}},
            {10,{256,84}},{11,{256,110}},{12,{256,144}},{13,{256,176}},{14,{256,210}},{15,{256,292}},
            {16,{256,66}},{17,{256,74}},{18,{256,98}},{19,{256,50}},{20,{32,18}},{21,{256,110}},
            {22,{256,82}},{23,{256,136}},{24,{1,1}},{25,{1,2}},{26,{1,4}},{27,{1,8}},{28,{1,8}},
            {29,{256,56}},{30,{1,2}},{34,{256,54}},{35,{256,66}},{39,{32,17}},{40,{64,36}},{41,{128,18}}
        };
        for (uint64_t i = 0; i < tensors; ++i) {
            const auto name = string();
            tensor_names.push_back(name);
            const auto unique = names.insert(name);
            if (!utf8(name) || !unique.second) throw std::runtime_error("Invalid GGUF tensor name");
            const auto dimensions = number(4);
            if (dimensions > remaining() / 8) throw std::runtime_error("Invalid GGUF dimensions");
            uint64_t elements = 1, row = 0;
            for (uint64_t j = 0; j < dimensions; ++j) {
                const auto extent = number(8); if (!j) row = extent;
                if (extent && elements > std::numeric_limits<uint64_t>::max() / extent) throw std::runtime_error("GGUF tensor too large");
                elements *= extent;
            }
            const auto type = number(4), offset = number(8);
            const auto storage = sizes.at(type);
            const bool raw = type == 0 || type == 1 || (type >= 24 && type <= 28);
            if (!raw && (!dimensions || row % storage.first)) throw std::runtime_error("Invalid GGUF quantized row");
            if (elements > std::numeric_limits<uint64_t>::max() / storage.second) throw std::runtime_error("GGUF tensor too large");
            data.push_back({offset, elements * storage.second / storage.first});
        }
        const auto padding = (alignment - offset_ % alignment) % alignment;
        if (offset_ > std::numeric_limits<uint64_t>::max() - padding) throw std::runtime_error("GGUF offset overflow");
        const auto start = offset_ + padding;
        for (const auto &tensor : data) {
            if (tensor.bytes && (start > size_ || tensor.offset > size_ - start || tensor.bytes > size_ - start - tensor.offset)) throw std::runtime_error("Truncated GGUF tensor");
        }
        return result;
    }
};
}
EmbeddingMetadata embedding_model_metadata(const std::string &path) {
    EmbeddingMetadata result;
    try {
        const auto fields = Gguf(path).fields();
        std::vector<std::string> prefixes;
        const auto architecture = fields.find("general.architecture");
        if (architecture != fields.end() && architecture->second.truthy()) prefixes.push_back(architecture->second.architecture());
        prefixes.insert(prefixes.end(), {"qwen3", "qwen2", "llama"});
        for (const auto &prefix : prefixes) {
            const std::pair<const char *, int64_t *> keys[] = {{".block_count", &result.layers}, {".attention.head_count_kv", &result.kv_heads}, {".attention.key_length", &result.key_length}, {".attention.value_length", &result.value_length}};
            bool found = false;
            for (const auto &key : keys) { const auto item = fields.find(prefix + key.first); if (item != fields.end() && item->second.truthy()) { *key.second = item->second.integer(); found = true; } }
            if (found) break;
        }
    } catch (const std::exception &) { /* Reference retains assignments preceding a conversion failure. */ }
    return result;
}
int64_t hybrid_llm_kv_bytes_per_token(const std::string &path, const std::string &raw_cache) {
    const auto cache = lower(strip_config_value(raw_cache.empty() ? "fp16" : raw_cache));
    static std::mutex mutex;
    static std::map<std::pair<std::string, std::string>, int64_t> cached;
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(path, cache);
    if (const auto found = cached.find(key); found != cached.end()) return found->second;
    int64_t heads = 4, key_length = 256, value_length = 256, layers = 0;
    try {
        Gguf reader(path);
        const auto fields = reader.fields();
        std::vector<std::string> prefixes;
        const auto architecture = fields.find("general.architecture");
        if (architecture != fields.end() && architecture->second.truthy()) prefixes.push_back(architecture->second.architecture());
        prefixes.insert(prefixes.end(), {"qwen35", "qwen35moe"});
        std::set<std::string> visited;
        for (const auto &prefix : prefixes) {
            if (!visited.insert(prefix).second) continue;
            const auto get = [&](const char *suffix) {
                const auto found = fields.find(prefix + suffix);
                return found == fields.end() ? Value{} : found->second;
            };
            const auto blocks = get(".block_count"), interval = get(".full_attention_interval"), kv = get(".attention.head_count_kv");
            const auto k = get(".attention.key_length"), v = get(".attention.value_length");
            if (kv.truthy()) heads = kv.integer();
            if (k.truthy()) key_length = k.integer();
            if (v.truthy()) value_length = v.integer();
            if (blocks.truthy() && interval.truthy()) {
                const auto divisor = interval.integer();
                if (!divisor) throw std::runtime_error("Zero GGUF attention interval");
                layers = std::max<int64_t>(1, static_cast<int64_t>(std::floor(static_cast<long double>(blocks.integer()) / divisor)));
            }
            if (kv.truthy() || blocks.truthy()) break;
        }
        if (!layers) for (const auto &name : reader.tensor_names) if (name.find(".attn_output.weight") != std::string::npos) ++layers;
    } catch (const std::exception &) { /* Keep earlier metadata assignments, as Pipes does. */ }
    if (!layers) layers = 16;
    const double bytes = cache == "q4_0" ? .5625 : cache == "q8_0" ? 1.0625 : cache == "f32" || cache == "fp32" ? 4.0 : 2.0;
    const long double result = std::ceil((static_cast<long double>(key_length) + value_length) * heads * layers * bytes);
    if (!std::isfinite(result) || result < std::numeric_limits<int64_t>::min() || result > std::numeric_limits<int64_t>::max()) throw std::runtime_error("GGUF KV size outside native range");
    return cached[key] = static_cast<int64_t>(result);
}
namespace {
// xllamacpp 2026.10.11549 returns MemoryEstimate without a `memory` field.
// Its allocation arithmetic is discarded by Pipes. Validate the metadata
// operations that can fail before that return, without allocating a KV vector
// or mapping tensor weights just to discard the computed allocation.
void validate_llm_memory_estimator(const std::string &path, const std::vector<std::string> &projectors) {
    for (const auto &projector : projectors) Gguf(projector).fields();
    Gguf reader(path, true);
    reader.fields();
    const auto &metadata = reader.metadata;
    const auto architecture = metadata.at("general.architecture").get<std::string>();
    const auto scalar = [](const nlohmann::json &value) -> double {
        if (value.is_boolean()) return value.get<bool>();
        if (!value.is_number()) throw std::runtime_error("Non-numeric GGUF graph metadata");
        const double number = value.get<double>();
        if (!std::isfinite(number)) throw std::runtime_error("Non-finite GGUF graph metadata");
        return number;
    };
    const auto extrema = [&](const nlohmann::json &value) {
        if (!value.is_array()) { const auto n = scalar(value); return std::make_pair(n, n); }
        if (value.empty()) throw std::runtime_error("Empty GGUF head counts");
        double high = scalar(value[0]), low = high;
        for (const auto &item : value) { const double n = scalar(item); high = std::max(high, n); low = std::min(low, n); }
        return std::make_pair(high, low);
    };
    scalar(metadata.at(architecture + ".embedding_length"));
    const auto &blocks = metadata.at(architecture + ".block_count");
    if (!blocks.is_number_integer() && !blocks.is_boolean()) throw std::runtime_error("Non-integer GGUF block count");
    const int64_t count = blocks.is_boolean() ? int(blocks.get<bool>()) : blocks.get<int64_t>();
    extrema(metadata.value(architecture + ".attention.head_count", nlohmann::json(1)));
    extrema(metadata.value(architecture + ".attention.head_count_kv", nlohmann::json(1)));
    const auto &tokens = metadata.at("tokenizer.ggml.tokens");
    if (!tokens.is_array() && !tokens.is_string()) throw std::runtime_error("GGUF vocabulary has no length");
    for (const auto *suffix : {".attention.key_length", ".attention.value_length"})
        if (metadata.contains(architecture + suffix)) scalar(metadata.at(architecture + suffix));
    if (architecture == "gemma3") scalar(metadata.at(architecture + ".attention.sliding_window"));
    for (const auto &name : reader.tensor_names) if (name.rfind("blk.", 0) == 0) {
        const auto end = name.find('.', 4);
        const auto raw = name.substr(4, end == std::string::npos ? end : end - 4);
        const auto parsed = parse_configuration_integer(raw);
        if (!parsed) throw std::runtime_error("Invalid GGUF block index");
        int64_t index = std::stoll(*parsed);
        if (index < 0) index += std::max<int64_t>(0, count);
        if (index < 0 || index >= count) throw std::runtime_error("GGUF block index outside model");
    }
}
}
double estimate_llm_memory_gb(const std::string &path, int64_t context,
    const std::vector<std::string> &projectors, const std::string &cache) {
    const auto name = lower(path);
    const bool hybrid = name.find("qwen3.5") != std::string::npos || name.find("qwen3.6") != std::string::npos || name.find("qwen3.8") != std::string::npos;
    const auto fallback = [&] {
        double weights = 4.0;
        try { if (!path.empty() && std::filesystem::exists(path)) weights = std::filesystem::file_size(path) / gib * (hybrid ? 1.05 : 1.2); } catch (const std::exception &) {}
        const double kv = hybrid ? context * static_cast<double>(hybrid_llm_kv_bytes_per_token(path, cache)) / gib
            : context / 1000.0 * std::min(.14 * (weights / 2.5), .1);
        return weights + kv + (hybrid ? .565 : 1.0);
    };
    try { validate_llm_memory_estimator(path, projectors); }
    catch (const std::exception &) {
        try {
            if (!path.empty() && std::filesystem::exists(path)) {
                const double weights = std::filesystem::file_size(path) / gib;
                const double kv = context * static_cast<double>(hybrid ? hybrid_llm_kv_bytes_per_token(path, cache) : 6000) / gib;
                double projector_weights = 0;
                for (const auto &projector : projectors) {
                    try { if (!projector.empty() && std::filesystem::exists(projector)) projector_weights += std::filesystem::file_size(projector) / gib; }
                    catch (const std::exception &) { projector_weights += .86; }
                }
                return weights + kv + projector_weights + .5;
            }
        } catch (const std::exception &) {}
    }
    return fallback();
}
double embedding_cache_bytes(const std::string &cache) {
    const auto type = lower(strip_config_value(cache));
    if (type == "f32" || type == "float32") return 4;
    if (type == "q8_0" || type == "q8") return 1;
    if (type == "q4_0" || type == "q4") return .5;
    return 2;
}
EmbeddingMemory estimate_embedding_memory(const EmbeddingMetadata &m, uint64_t bytes, int context, int parallel, int ubatch, const std::string &cache) {
    EmbeddingMemory out; out.layers = std::max<int64_t>(1, m.layers); out.weights_gb = bytes / gib;
    const double kv_layer = (static_cast<double>(std::max(1, context)) * std::max(1, parallel) * m.kv_heads * (static_cast<double>(m.key_length) + m.value_length) * embedding_cache_bytes(cache)) / gib;
    out.compute_gb = .25 + std::max(1, ubatch) * .0007;
    out.fixed_gb = out.weights_gb * .1 + out.compute_gb;
    out.per_layer_gb = out.weights_gb * .9 / out.layers + kv_layer;
    out.kv_gb = kv_layer * out.layers;
    out.total_gb = out.fixed_gb + out.per_layer_gb * out.layers;
    return out;
}
EmbeddingMemory estimate_embedding_memory(const std::string &path, int context, int parallel, int ubatch, const std::string &cache) {
    return estimate_embedding_memory(embedding_model_metadata(path), std::filesystem::file_size(path), context, parallel, ubatch, cache);
}
int resolve_embedding_gpu_layers(const EmbeddingMemory &m, std::optional<int> requested, bool force_cpu, bool cuda, double free_gb, double safety_gb) {
    if (force_cpu || !cuda) return 0;
    if (requested) return *requested;
    safety_gb = std::max(0.0, safety_gb);
    if (free_gb >= m.total_gb + safety_gb) return -1;
    const double selected = std::trunc((free_gb - safety_gb - m.fixed_gb) / std::max(.001, m.per_layer_gb));
    if (!std::isfinite(selected)) throw std::runtime_error("Cannot convert non-finite embedding layer budget to integer");
    if (selected <= 0) return 0;
    if (selected >= m.layers) return -1;
    if (selected > std::numeric_limits<int>::max()) throw std::runtime_error("Embedding layer count exceeds native engine range");
    return static_cast<int>(selected);
}
std::vector<int> embedding_fallback_layers(int selected, int64_t layers) {
    if (!selected) return {0};
    const auto start = selected < 0 ? layers : std::min<int64_t>(selected, layers);
    std::vector<int> out{selected};
    for (const double fraction : {.75, .5, .25}) {
        const double scaled = std::max(1.0, std::trunc(start * fraction));
        if (scaled > std::numeric_limits<int>::max()) throw std::runtime_error("Embedding layer count exceeds native engine range");
        const auto candidate = static_cast<int>(scaled);
        if (std::find(out.begin(), out.end(), candidate) == out.end()) out.push_back(candidate);
    }
    out.push_back(0); return out;
}
double embedding_gpu_memory(const EmbeddingMemory &m, int selected) {
    return selected == 0 ? 0 : m.fixed_gb + m.per_layer_gb * (selected < 0 ? m.layers : std::min<int64_t>(selected, m.layers));
}
bool embedding_memory_error(std::string message) {
    message = lower(std::move(message));
    for (const auto *token : {"out of memory", "cuda", "vram", "alloc", "memory", "resource"}) if (message.find(token) != std::string::npos) return true;
    return false;
}
std::vector<std::string> embedding_warmup_inputs(bool cuda, int context, int ubatch) {
    if (!cuda || lower(strip_config_value(ezl_getenv_fallback("EMBEDDING_KEEP_LOADED", "true"))) != "true" || lower(strip_config_value(ezl_getenv_fallback("EMBEDDING_WARMUP", "true"))) != "true") return {};
    std::vector<std::string> inputs;
    if (context >= 16) inputs.push_back("\"Warm up the embedding model.\"");
    const int tokens = std::max(1, std::min(ubatch, context));
    std::string input = "[0"; for (int i = 1; i < tokens; ++i) input += ",0";
    inputs.push_back(input + ']'); return inputs;
}
}
