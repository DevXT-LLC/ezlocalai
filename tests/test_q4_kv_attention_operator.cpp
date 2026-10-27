// GPU-only oracle for the actual ggml FlashAttention operator. A maintenance
// controller must own GPU0; no device is opened without --execute-on-gpu0.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "fattn-q4-kv-policy.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr int64_t dimension = 256;
constexpr int64_t kv_heads = 4;
constexpr int64_t dictionary_rows = 1024;
constexpr int64_t mask_rows = 32;
constexpr const char * fusion_environment = "GGML_CUDA_FATTN_Q4_KV_FUSION";
constexpr size_t mib = 1024 * 1024;
void check(bool value, const std::string & message) { if (!value) throw std::runtime_error(message); }

struct Diagnostics {
    using Reset = void (*)();
    using Get = void (*)(ggml_cuda_fattn_q4_kv_stats *);
    Reset reset = nullptr;
    Get get = nullptr;
    bool available() const { return reset != nullptr && get != nullptr; }
    bool absent() const { return reset == nullptr && get == nullptr; }
    void clear() const { if (reset) reset(); }
    void read(ggml_cuda_fattn_q4_kv_stats & value) const { if (get) get(&value); else value = {}; }
};
template<typename Function> Function diagnostic_symbol(const char * name) {
    void * address = dlsym(RTLD_DEFAULT, name);
    Function function = nullptr;
    static_assert(sizeof(function) == sizeof(address), "POSIX diagnostic function pointer size");
    std::memcpy(&function, &address, sizeof(function));
    return function;
}
Diagnostics load_diagnostics() {
    Diagnostics result;
    result.reset = diagnostic_symbol<Diagnostics::Reset>("ggml_cuda_fattn_q4_kv_reset_stats");
    result.get = diagnostic_symbol<Diagnostics::Get>("ggml_cuda_fattn_q4_kv_get_stats");
    check(result.available() || result.absent(), "partial Q4 diagnostic ABI: both exports or neither are required");
    return result;
}

struct Environment {
    std::string name, value;
    bool present;
    explicit Environment(const char * key) : name(key), present(std::getenv(key) != nullptr) {
        if (present) value = std::getenv(key);
    }
    ~Environment() { if (present) setenv(name.c_str(), value.c_str(), 1); else unsetenv(name.c_str()); }
};

uint32_t mix(uint32_t value) {
    value ^= value >> 16; value *= 0x7feb352dU;
    value ^= value >> 15; value *= 0x846ca68bU;
    return value ^ (value >> 16);
}
float random_finite(uint32_t value) {
    return (static_cast<float>(mix(value) & 0x00ffffffU) / 8388608.0f) - 1.0f;
}
std::string fingerprint(const void * data, size_t bytes) {
    const auto * input = static_cast<const uint8_t *>(data);
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes; ++i) { hash ^= input[i]; hash *= 1099511628211ULL; }
    std::ostringstream out; out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

struct Case {
    int64_t kv = 256, queries = 4, heads = 24, offset_tokens = 0, active_tokens = 0, sequences = 1;
    std::string layout = "token_major", pattern = "random", exclusion;
    ggml_type type = GGML_TYPE_Q4_0;
    bool with_mask = true, with_sinks = false, shared_kv = false, expected_fused = true;
    float max_bias = 0, softcap = 0;
    int32_t requested_sparse = 0;
};
std::string name(const Case & c) {
    return c.pattern + ":" + c.layout + ":kv" + std::to_string(c.kv) + ":q" +
        std::to_string(c.queries) + ":h" + std::to_string(c.heads) + ":offset" +
        std::to_string(c.offset_tokens) + ":s" + std::to_string(c.sequences) + (c.exclusion.empty() ? "" : ":excluded_" + c.exclusion);
}
size_t estimate_device_bytes(const Case & c) {
    const size_t storage_rows = static_cast<size_t>(c.kv + 2 * c.offset_tokens) * kv_heads * c.sequences;
    const size_t packed = storage_rows * ggml_row_size(c.type, dimension) * 2;
    // Keep the baseline's full-F16 K/V allocation in the conservative estimate.
    // This excludes CUDA context/driver memory, which is reported separately.
    const size_t f16 = storage_rows * dimension * sizeof(ggml_fp16_t) * 2;
    return packed + f16 + static_cast<size_t>(c.kv) * mask_rows * c.sequences * sizeof(ggml_fp16_t) +
        static_cast<size_t>(dimension * c.heads * c.queries * c.sequences) * sizeof(float) * 2 + 64 * mib;
}
std::vector<Case> cases(const std::string & suite, bool near_context) {
    std::vector<Case> result;
    const std::vector<int64_t> lengths = suite == "quick" ? std::vector<int64_t>{256, 8192} :
        std::vector<int64_t>{256, 512, 8192, 130048, 130304};
    const std::vector<std::string> patterns = suite == "quick" ? std::vector<std::string>{"random", "alternating"} :
        std::vector<std::string>{"random", "zero", "alternating", "scaled"};
    for (int64_t length : lengths) for (int64_t queries : {3, 4})
        for (const auto & layout : {"head_major", "token_major"}) for (const auto & pattern : patterns) {
            Case c; c.kv = length; c.queries = queries; c.layout = layout; c.pattern = pattern;
            c.active_tokens = length - 7;
            result.push_back(c);
        }
    if (near_context) for (int64_t queries : {3, 4}) {
        Case c; c.kv = 230144; c.active_tokens = 230000; c.queries = queries;
        result.push_back(c);
    }
    // A token-major subview still has a dense logical span; guard tokens must remain unchanged.
    for (int64_t queries : {3, 4}) {
        Case c; c.queries = queries; c.offset_tokens = 1; c.active_tokens = c.kv - 7; result.push_back(c);
    }
    // Excluded cases must preserve the stock operator, including its output bits.
    for (const auto & exclusion : {"gqa4", "gqa8", "two_queries", "f16_kv", "no_mask", "alibi", "softcap", "sinks", "shared_kv", "two_sequences", "requested_sparse", "head_view_gap"}) {
        Case c; c.active_tokens = c.kv - 7; c.expected_fused = false; c.exclusion = exclusion;
        if (c.exclusion == "gqa4") c.heads = 16;
        if (c.exclusion == "gqa8") c.heads = 32;
        if (c.exclusion == "two_queries") c.queries = 2;
        if (c.exclusion == "f16_kv") c.type = GGML_TYPE_F16;
        if (c.exclusion == "no_mask") c.with_mask = false;
        if (c.exclusion == "alibi") c.max_bias = 0.125f;
        if (c.exclusion == "softcap") c.softcap = 10.0f;
        if (c.exclusion == "sinks") c.with_sinks = true;
        if (c.exclusion == "shared_kv") c.shared_kv = true;
        if (c.exclusion == "two_sequences") c.sequences = 2;
        if (c.exclusion == "requested_sparse") c.requested_sparse = static_cast<int32_t>(c.kv);
        if (c.exclusion == "head_view_gap") { c.layout = "head_major"; c.offset_tokens = 1; }
        result.push_back(c);
    }
    return result;
}

float dictionary_value(const Case & c, uint32_t row, uint32_t channel, bool value) {
    const uint32_t tag = value ? 0x50607080U : 0x10203040U;
    if (c.pattern == "zero") return 0.0f;
    if (c.pattern == "alternating") return ((row + channel) & 1U ? -1.0f : 1.0f) * (value ? 16.0f : 0.25f);
    const float ordinary = random_finite(tag ^ row * 257U ^ channel * 65537U);
    if (c.pattern != "scaled") return ordinary * (value ? 2.0f : 0.5f);
    constexpr float scales[] = {0x1p-16f, 0x1p-8f, 0.125f, 1.0f, 16.0f};
    return ordinary * scales[(row + channel / 32U) % 5U];
}
std::vector<uint8_t> packed_dictionary(const Case & c, bool value) {
    std::vector<float> source(dictionary_rows * dimension);
    for (uint32_t row = 0; row < dictionary_rows; ++row) for (uint32_t col = 0; col < dimension; ++col)
        source[row * dimension + col] = dictionary_value(c, row, col, value);
    std::vector<uint8_t> result(dictionary_rows * ggml_row_size(c.type, dimension));
    if (c.type == GGML_TYPE_Q4_0) {
        const auto bytes = ggml_quantize_chunk(c.type, source.data(), result.data(), 0, dictionary_rows, dimension, nullptr);
        check(bytes == result.size(), "actual ggml Q4 quantizer returned a different byte count");
    } else {
        check(c.type == GGML_TYPE_F16, "unsupported oracle KV type");
        ggml_fp32_to_fp16_row(source.data(), reinterpret_cast<ggml_fp16_t *>(result.data()), source.size());
    }
    return result;
}
struct Input {
    std::vector<float> q, sinks;
    std::vector<uint8_t> k, v;
    std::vector<ggml_fp16_t> mask;
};
Input prepare(const Case & c) {
    Input input;
    const size_t row_bytes = ggml_row_size(c.type, dimension);
    const int64_t storage_tokens = c.kv + 2 * c.offset_tokens;
    const size_t storage_rows = static_cast<size_t>(storage_tokens) * kv_heads * c.sequences;
    input.k.assign(storage_rows * row_bytes, 0xa5); input.v.assign(storage_rows * row_bytes, 0x5a);
    const auto keys = packed_dictionary(c, false), values = packed_dictionary(c, true);
    for (int64_t sequence = 0; sequence < c.sequences; ++sequence)
      for (int64_t head = 0; head < kv_heads; ++head) for (int64_t token = 0; token < c.kv; ++token) {
        const auto dictionary = mix(static_cast<uint32_t>(token) ^ static_cast<uint32_t>(head * 0x10001) ^ static_cast<uint32_t>(sequence * 0x998877)) % dictionary_rows;
        const auto row = sequence * storage_tokens * kv_heads + (c.layout == "head_major" ?
            head * storage_tokens + token + c.offset_tokens : (token + c.offset_tokens) * kv_heads + head);
        std::memcpy(input.k.data() + row * row_bytes, keys.data() + dictionary * row_bytes, row_bytes);
        std::memcpy(input.v.data() + row * row_bytes, values.data() + dictionary * row_bytes, row_bytes);
    }
    if (c.shared_kv) input.v = input.k;
    input.q.resize(dimension * c.queries * c.heads * c.sequences);
    for (int64_t sequence = 0; sequence < c.sequences; ++sequence) for (int64_t head = 0; head < c.heads; ++head) for (int64_t query = 0; query < c.queries; ++query)
        for (int64_t channel = 0; channel < dimension; ++channel) {
            const auto logical = static_cast<uint32_t>(((sequence * c.heads + head) * c.queries + query) * dimension + channel);
            const auto row = sequence * c.heads * c.queries + (c.layout == "head_major" ? head * c.queries + query : query * c.heads + head);
            input.q[row * dimension + channel] = c.pattern == "zero" ? 0.0f : c.pattern == "alternating" ?
                (logical & 1U ? -4.0f : 4.0f) : random_finite(logical ^ 0x99887766U) * 0.5f;
        }
    if (c.with_mask) {
        const int64_t active = c.active_tokens > 0 ? c.active_tokens : c.kv;
        check(active >= c.queries && active <= c.kv, "invalid active/padded query geometry");
        input.mask.assign(c.kv * mask_rows * c.sequences, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t sequence = 0; sequence < c.sequences; ++sequence)
          for (int64_t q = 0; q < c.queries; ++q) for (int64_t k = 0; k <= active - c.queries + q; ++k)
            input.mask[(sequence * mask_rows + q) * c.kv + k] = ggml_fp32_to_fp16(0.0f);
    }
    if (c.with_sinks) input.sinks.assign(c.heads, 0.0f);
    return input;
}

Json stats_json(const ggml_cuda_fattn_q4_kv_stats & s) {
#define FIELD(name) {#name, s.name}
    return Json{FIELD(eligible_calls), FIELD(fused_calls), FIELD(occupancy_fallbacks),
        FIELD(baseline_blocks_per_sm), FIELD(candidate_blocks_per_sm),
        FIELD(grid_x), FIELD(grid_y), FIELD(grid_z), FIELD(block_x), FIELD(block_y), FIELD(block_z),
        FIELD(shared_bytes), FIELD(kv_rows), FIELD(query_rows), FIELD(tile_rows), FIELD(output_tiles), FIELD(stream_k)};
#undef FIELD
}
bool all_stats_zero(const ggml_cuda_fattn_q4_kv_stats & stats) {
    const Json values = stats_json(stats);
    for (const auto & value : values) if (value.get<uint64_t>() != 0) return false;
    return true;
}
enum class Mode { off, on, automatic };
void same_partition(const ggml_cuda_fattn_q4_kv_stats & a, const ggml_cuda_fattn_q4_kv_stats & b) {
#define SAME(name) check(a.name == b.name, "stock/fused launch partition differs: " #name)
    SAME(grid_x); SAME(grid_y); SAME(grid_z); SAME(block_x); SAME(block_y); SAME(block_z);
    SAME(shared_bytes); SAME(kv_rows); SAME(query_rows); SAME(tile_rows); SAME(output_tiles); SAME(stream_k);
    SAME(baseline_blocks_per_sm);
#undef SAME
    check(b.candidate_blocks_per_sm >= b.baseline_blocks_per_sm, "fused occupancy cannot sustain the stock launch partition");
}

struct Variant {
    std::vector<float> output;
    ggml_cuda_fattn_q4_kv_stats stats{};
    std::vector<double> compute_wall_ms;
    size_t tensor_buffer_bytes = 0, retained_device_delta_bytes = 0;
    bool unused_workspace_guard_checked = false;
    size_t unused_workspace_guard_bytes = 0;
    bool workspace_probe_checked = false;
    size_t workspace_probe_bytes = 0, workspace_probe_changed_bytes = 0;
    std::string workspace_probe_expected = "not_installed", workspace_probe_observed = "not_checked";
};
Variant execute(const Case & c, const Input & input, Mode mode, int repeats,
        const Diagnostics & diagnostics, bool stock_control, bool auto_control, bool auto_twin_proven = false) {
    const bool fused = mode == Mode::on;
    const bool automatic = mode == Mode::automatic;
    check(!automatic || (auto_control && !stock_control), "AUTO requires explicit normal-backend auto-control");
    check(automatic ? unsetenv(fusion_environment) == 0 : setenv(fusion_environment, fused ? "1" : "0", 1) == 0,
          "set OFF/ON/AUTO environment mode");
    size_t free_before = 0, total = 0;
    ggml_backend_cuda_get_device_memory(0, &free_before, &total);
    Context context(ggml_init({8 * mib, nullptr, true}), ggml_free);
    check(bool(context), "allocate ggml tensor metadata");
    Backend backend(ggml_backend_cuda_init(0), ggml_backend_free);
    check(bool(backend) && ggml_backend_is_cuda(backend.get()), "actual CUDA backend initialization");
    auto * q_storage = c.layout == "head_major" ?
        ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, dimension, c.queries, c.heads, c.sequences) :
        ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, dimension, c.heads, c.queries, c.sequences);
    auto * q = c.layout == "head_major" ? q_storage : ggml_permute(context.get(), q_storage, 0, 2, 1, 3);
    const int64_t storage_tokens = c.kv + 2 * c.offset_tokens;
    auto make_kv = [&](const char * label) {
        auto * base = c.layout == "head_major" ?
            ggml_new_tensor_4d(context.get(), c.type, dimension, storage_tokens, kv_heads, c.sequences) :
            ggml_new_tensor_4d(context.get(), c.type, dimension, kv_heads, storage_tokens, c.sequences);
        ggml_set_name(base, label);
        auto * t = c.layout == "head_major" ? base : ggml_permute(context.get(), base, 0, 2, 1, 3);
        if (c.offset_tokens) {
            const size_t offset = c.layout == "head_major" ? c.offset_tokens * base->nb[1] : c.offset_tokens * base->nb[2];
            t = ggml_view_4d(context.get(), t, dimension, c.kv, kv_heads, c.sequences, t->nb[1], t->nb[2], t->nb[3], offset);
        }
        return std::make_pair(base, t);
    };
    const auto k = make_kv("k_storage");
    const auto v = c.shared_kv ? std::make_pair(k.first, ggml_view_4d(context.get(), k.second, dimension, c.kv, kv_heads, c.sequences, k.second->nb[1], k.second->nb[2], k.second->nb[3], 0)) : make_kv("v_storage");
    auto * mask = c.with_mask ? ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, c.kv, mask_rows, 1, c.sequences) : nullptr;
    auto * sinks = c.with_sinks ? ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, c.heads) : nullptr;
    auto * output = ggml_flash_attn_ext(context.get(), q, k.second, v.second, mask, 1.0f / std::sqrt(float(dimension)), c.max_bias, c.softcap);
    ggml_flash_attn_ext_add_sinks(output, sinks);
    ggml_flash_attn_ext_set_n_kv_max(output, c.requested_sparse);
    ggml_prec_set_acc(output, GGML_PREC_F32);
    ggml_set_input(q_storage); ggml_set_input(k.first); ggml_set_input(v.first);
    if (mask) ggml_set_input(mask);
    if (sinks) ggml_set_input(sinks);
    ggml_set_output(output);
    check(ggml_backend_supports_op(backend.get(), output), "CUDA backend rejects the actual FA operator");
    auto * graph = ggml_new_graph_custom(context.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    check(bool(buffer), "allocate actual GPU graph tensors and F16 workspace");
    ggml_backend_tensor_set(q_storage, input.q.data(), 0, input.q.size() * sizeof(float));
    ggml_backend_tensor_set(k.first, input.k.data(), 0, input.k.size());
    ggml_backend_tensor_set(v.first, input.v.data(), 0, input.v.size());
    if (mask) ggml_backend_tensor_set(mask, input.mask.data(), 0, input.mask.size() * sizeof(ggml_fp16_t));
    if (sinks) ggml_backend_tensor_set(sinks, input.sinks.data(), 0, input.sinks.size() * sizeof(float));
    diagnostics.clear();
    check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "warm actual FA graph");
    ggml_backend_synchronize(backend.get());
    Variant result;
    result.tensor_buffer_bytes = ggml_backend_buffer_get_size(buffer.get());
    result.output.resize(ggml_nelements(output));
    std::vector<float> warm(result.output.size());
    ggml_backend_tensor_get(output, warm.data(), 0, warm.size() * sizeof(float));
    unsigned char * workspace_probe = nullptr;
    const bool probe_overwritten = auto_control && mode == Mode::off;
    if (!stock_control && c.expected_fused && (fused || automatic || probe_overwritten)) {
        ggml_cuda_fattn_q4_kv_stats warm_stats{};
        diagnostics.read(warm_stats);
        if (automatic) {
            check(auto_twin_proven, "AUTO workspace guard requires the completed same-input ON twin proof");
            check(all_stats_zero(warm_stats), "AUTO warmup unexpectedly recorded diagnostic stats");
        } else {
            check(warm_stats.eligible_calls == 1 && warm_stats.fused_calls == (fused ? 1U : 0U) &&
                  warm_stats.occupancy_fallbacks == 0, "warmup did not select the expected OFF/ON path for its workspace probe");
        }
        constexpr size_t guard_bytes = 256;
        const size_t logical = ggml_nbytes(output);
        const size_t allocated = ggml_backend_buffer_get_alloc_size(buffer.get(), output);
        check(allocated >= logical + guard_bytes, "baseline F16 workspace reservation is missing after output");
        const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer.get()));
        const uintptr_t data = reinterpret_cast<uintptr_t>(output->data);
        check(data >= base && data - base <= ggml_backend_buffer_get_size(buffer.get()) &&
              logical + guard_bytes <= ggml_backend_buffer_get_size(buffer.get()) - (data - base),
              "workspace probe would exceed the actual GPU buffer");
        workspace_probe = static_cast<unsigned char *>(output->data) + logical;
        check(cudaMemset(workspace_probe, 0xa5, guard_bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
              "initialize F16-workspace probe");
        result.workspace_probe_checked = true;
        result.workspace_probe_bytes = guard_bytes;
        result.workspace_probe_expected = probe_overwritten ? "overwritten" : "preserved";
        // OFF deliberately uses this region; only ON/AUTO have an unused-tail guard.
        result.unused_workspace_guard_checked = !probe_overwritten;
        result.unused_workspace_guard_bytes = probe_overwritten ? 0 : guard_bytes;
    }
    diagnostics.clear();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto before = std::chrono::steady_clock::now();
        check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "compute actual FA graph");
        ggml_backend_synchronize(backend.get());
        const auto after = std::chrono::steady_clock::now();
        result.compute_wall_ms.push_back(std::chrono::duration<double, std::milli>(after - before).count());
        ggml_backend_tensor_get(output, result.output.data(), 0, result.output.size() * sizeof(float));
        check(std::memcmp(warm.data(), result.output.data(), warm.size() * sizeof(float)) == 0,
              "a fixed actual FA graph changed output bytes after warmup/repeat");
    }
    diagnostics.read(result.stats);
    if (workspace_probe) {
        std::vector<unsigned char> guard(result.workspace_probe_bytes);
        check(cudaMemcpy(guard.data(), workspace_probe, guard.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
              "read F16-workspace probe");
        result.workspace_probe_changed_bytes = static_cast<size_t>(std::count_if(guard.begin(), guard.end(),
            [](unsigned char byte) { return byte != 0xa5; }));
        const bool changed = result.workspace_probe_changed_bytes != 0;
        result.workspace_probe_observed = changed ? "overwritten" : "preserved";
        check(probe_overwritten ? changed : !changed, probe_overwritten ?
            "OFF did not overwrite the F16-workspace positive control" :
            "ON/AUTO wrote past the logical FA output into unused F16 workspace");
    }
    check(std::all_of(result.output.begin(), result.output.end(), [](float f) { return std::isfinite(f); }),
          "finite inputs produced a nonfinite actual FA output");
    auto unchanged = [](ggml_tensor * tensor, const void * expected, size_t bytes, const char * field) {
        std::vector<uint8_t> after(bytes); ggml_backend_tensor_get(tensor, after.data(), 0, bytes);
        check(std::memcmp(after.data(), expected, bytes) == 0, std::string("operator modified input/guard bytes: ") + field);
    };
    unchanged(q_storage, input.q.data(), input.q.size() * sizeof(float), "Q");
    unchanged(k.first, input.k.data(), input.k.size(), "K");
    unchanged(v.first, input.v.data(), input.v.size(), "V");
    if (mask) unchanged(mask, input.mask.data(), input.mask.size() * sizeof(ggml_fp16_t), "mask");
    if (sinks) unchanged(sinks, input.sinks.data(), input.sinks.size() * sizeof(float), "sinks");
    size_t free_after = 0; ggml_backend_cuda_get_device_memory(0, &free_after, &total);
    result.retained_device_delta_bytes = free_before > free_after ? free_before - free_after : 0;
    if (automatic) {
        check(all_stats_zero(result.stats), "AUTO recorded diagnostic stats instead of remaining unobserved");
    } else if (!stock_control) {
        if (c.expected_fused) {
            check(result.stats.eligible_calls == static_cast<uint64_t>(repeats), "eligible operator was not dispatched once per compute; check CUDA graphs/instrumentation");
            check(result.stats.fused_calls == (fused ? static_cast<uint64_t>(repeats) : 0), "requested stock/fused path was not actually used");
            check(result.stats.occupancy_fallbacks == 0, "fused path fell back because launch occupancy changed");
            check(result.stats.grid_x && result.stats.grid_y && result.stats.grid_z &&
                  result.stats.block_x && result.stats.block_y && result.stats.block_z &&
                  result.stats.tile_rows && result.stats.output_tiles && result.stats.shared_bytes &&
                  result.stats.baseline_blocks_per_sm, "actual launch/grid/occupancy proof is missing");
            check(result.stats.kv_rows == static_cast<uint64_t>(c.kv) &&
                  result.stats.query_rows == static_cast<uint64_t>(c.queries), "launch proof used a different query/KV geometry");
        } else {
            check(result.stats.eligible_calls == 0 && result.stats.fused_calls == 0 && result.stats.occupancy_fallbacks == 0,
                  "excluded metadata unexpectedly selected the fused path");
        }
    }
    return result;
}

struct Options {
    bool execute = false, describe = false, near_context = false, help = false;
    bool stock_control = false, fused_only = false, auto_control = false;
    std::string suite = "quick";
    std::filesystem::path output;
    int repeats = 2, case_index = -1;
    size_t max_device_mib = 1024;
};
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&] { check(i + 1 < argc, "missing value for " + flag); return std::string(argv[++i]); };
        if (flag == "--help") o.help = true;
        else if (flag == "--execute-on-gpu0") o.execute = true;
        else if (flag == "--describe-cases") o.describe = true;
        else if (flag == "--near-context") o.near_context = true;
        else if (flag == "--stock-control") o.stock_control = true;
        else if (flag == "--auto-control") o.auto_control = true;
        else if (flag == "--fused-only") o.fused_only = true;
        else if (flag == "--suite") o.suite = value();
        else if (flag == "--output") o.output = value();
        else if (flag == "--repeats") o.repeats = std::stoi(value());
        else if (flag == "--case-index") o.case_index = std::stoi(value());
        else if (flag == "--max-device-mib") o.max_device_mib = std::stoul(value());
        else throw std::runtime_error("unknown option: " + flag);
    }
    check(o.suite == "quick" || o.suite == "full", "suite must be quick or full");
    check(o.repeats >= 1 && o.repeats <= 16 && o.case_index >= -1, "finite repeat/case bounds required");
    check(o.max_device_mib >= 128 && o.max_device_mib <= 2048, "estimated device allocation bound must be128..2048MiB");
    check(!o.execute || !o.output.empty(), "--output is required for GPU execution");
    check(!(o.execute && o.describe), "describe-cases and GPU execution are mutually exclusive");
    check(!(o.stock_control && o.fused_only), "stock-control and fused-only are mutually exclusive");
    check(!(o.stock_control && o.auto_control), "stock-control and auto-control are mutually exclusive");
    check(!o.auto_control || o.repeats == 2, "auto-control requires two measured repeats in each OFF/ON/AUTO context");
    check(!o.stock_control || (o.suite == "quick" && !o.near_context && o.case_index == -1 && o.repeats == 2),
          "stock-control requires the complete quick30 suite, two repeats and no near-context or case filter");
    check(!o.fused_only || o.case_index == -1, "fused-only requires the complete supported-geometry selection without a case filter");
    return o;
}
void save(const Options & o, const Json & report) {
    if (o.output.empty()) return;
    const auto temporary = o.output.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        check(bool(output), "open output report"); output << report.dump(2) << '\n';
        output.close(); check(bool(output), "write output report");
    }
    std::filesystem::rename(temporary, o.output);
}
} // namespace

int main(int argc, char ** argv) {
    Json report{{"schema_version", 1}, {"suite", "q4_kv_attention_operator"}, {"status", "not_run"},
        {"physical_gpu_execution", false}, {"diagnostic_exports_available", nullptr}, {"results", Json::array()}};
    Options options;
    try {
        options = parse(argc, argv);
        auto selected = cases(options.suite, options.near_context);
        if (options.fused_only) selected.erase(std::remove_if(selected.begin(), selected.end(),
            [](const Case & c) { return !c.expected_fused; }), selected.end());
        if (options.case_index >= 0) {
            check(static_cast<size_t>(options.case_index) < selected.size(), "case index outside fixed suite");
            selected = {selected[options.case_index]};
        }
        report.update({{"case_count", selected.size()}, {"selection", options.suite}, {"repeats", options.repeats},
            {"suite", options.stock_control ? "q4_kv_attention_stock_control" : "q4_kv_attention_operator"},
            {"stock_control", options.stock_control}, {"fused_only", options.fused_only},
            {"fusion_path_proof_required", !options.stock_control},
            {"comparison_context", options.stock_control ?
                "unchanged stock backend with OFF/ON environment values; no fusion path exists" :
                "experimental backend with stock OFF versus fused ON execution"},
            {"near_context_enabled", options.near_context}, {"max_estimated_device_mib", options.max_device_mib},
            {"device_estimate_excludes_cuda_context_and_driver", true}, {"dictionary_rows", dictionary_rows},
            {"kv_quantizer", "actual ggml_quantize_chunk Q4_0"}, {"case_index_filter", options.case_index}});
        if (options.auto_control) report.update({{"auto_control", true},
            {"comparison_context", "explicit OFF/ON diagnostic twins followed by a fresh environment-unset AUTO context"},
            {"guard_scope", "input guards remain preserved; OFF workspace is an overwritten positive control and ON/AUTO workspace guards remain preserved"},
            {"default_only_process_silence_proven", false}});
        if (options.describe) {
            for (const auto & c : selected) report["results"].push_back({{"name", name(c)},
                {"expected_fused", options.stock_control ? Json(nullptr) : Json(c.expected_fused)},
                {"candidate_geometry_supported", c.expected_fused},
                {"estimated_tensor_workspace_bytes", estimate_device_bytes(c)}});
            report["status"] = "described_without_gpu"; save(options, report); std::cout << report.dump(2) << '\n'; return 0;
        }
        if (!options.execute || options.help) {
            report["usage"] = "--execute-on-gpu0 --suite quick|full --output REPORT.json [--fused-only | --stock-control] [--auto-control] [--near-context --max-device-mib 2048]";
            std::cout << report.dump(2) << '\n'; return options.help ? 0 : 77;
        }
        for (const auto & c : selected) check(estimate_device_bytes(c) <= options.max_device_mib * mib,
            "fixed case exceeds allocation bound; explicitly raise --max-device-mib or choose a different fixed suite: " + name(c));
        const Diagnostics diagnostics = load_diagnostics();
        report["diagnostic_exports_available"] = diagnostics.available();
        check(options.stock_control ? diagnostics.absent() : diagnostics.available(), options.stock_control ?
            "stock-control requires both Q4 diagnostic exports absent; this is not the pristine stock backend" :
            "normal operator mode requires both Q4 diagnostic exports; stock controls must be explicitly requested");
        Environment original_fusion(fusion_environment), original_graphs("GGML_CUDA_DISABLE_GRAPHS");
        check(setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1) == 0, "disable graph replay for host path proof");
        int device_count = 0; check(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count == 1,
                                   "exactly one visible CUDA device is required");
        cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties, 0) == cudaSuccess, "read CUDA device properties");
        check(properties.major == 8 && properties.minor == 9, "this prototype oracle requires SM89");
        report.update({{"status", "running"}, {"physical_gpu_execution", true}, {"device", properties.name},
            {"compute_capability", properties.major * 10 + properties.minor}, {"cuda_graph_replay_disabled", true},
            {"ggml_version", ggml_version()}, {"ggml_commit", ggml_commit()}});
        save(options, report);
        for (size_t i = 0; i < selected.size(); ++i) {
            const auto & c = selected[i]; report["pending_case"] = name(c); save(options, report);
            const auto input = prepare(c);
            const Json input_fingerprints{{"q", fingerprint(input.q.data(), input.q.size() * sizeof(float))},
                {"k", fingerprint(input.k.data(), input.k.size())}, {"v", fingerprint(input.v.data(), input.v.size())},
                {"mask", fingerprint(input.mask.data(), input.mask.size() * sizeof(ggml_fp16_t))},
                {"sinks", fingerprint(input.sinks.data(), input.sinks.size() * sizeof(float))}};
            const auto baseline = execute(c, input, Mode::off, options.repeats, diagnostics, options.stock_control, options.auto_control);
            const auto fused = execute(c, input, Mode::on, options.repeats, diagnostics, options.stock_control, options.auto_control);
            check(baseline.output.size() == fused.output.size(), "OFF/ON output size differs");
            if (!options.stock_control && c.expected_fused) same_partition(baseline.stats, fused.stats);
            if (std::memcmp(baseline.output.data(), fused.output.data(), baseline.output.size() * sizeof(float)) != 0) {
                size_t first = 0;
                while (first < baseline.output.size() && std::memcmp(&baseline.output[first], &fused.output[first], sizeof(float)) == 0) ++first;
                uint32_t a = 0, b = 0; std::memcpy(&a, &baseline.output[first], sizeof(a)); std::memcpy(&b, &fused.output[first], sizeof(b));
                report["first_mismatch"] = {{"case", name(c)}, {"element", first}, {"baseline_bits", a}, {"candidate_bits", b},
                    {"baseline_value", baseline.output[first]}, {"candidate_value", fused.output[first]}};
                throw std::runtime_error("actual FA output differs at the bit level: " + name(c));
            }
            std::unique_ptr<Variant> automatic;
            if (options.auto_control) {
                const bool on_twin_proven = !c.expected_fused ||
                    (fused.stats.eligible_calls == static_cast<uint64_t>(options.repeats) &&
                     fused.stats.fused_calls == static_cast<uint64_t>(options.repeats) && fused.stats.occupancy_fallbacks == 0 &&
                     fused.unused_workspace_guard_checked && fused.workspace_probe_expected == "preserved" &&
                     fused.workspace_probe_observed == "preserved");
                automatic = std::make_unique<Variant>(execute(c, input, Mode::automatic, options.repeats,
                    diagnostics, false, true, on_twin_proven));
                check(automatic->output.size() == baseline.output.size(), "AUTO/OFF output size differs");
                if (std::memcmp(baseline.output.data(), automatic->output.data(), baseline.output.size() * sizeof(float)) != 0) {
                    report["first_mismatch"] = {{"case", name(c)}, {"comparison", "AUTO versus completed OFF/ON twins"}};
                    throw std::runtime_error("actual AUTO FA output differs at the bit level: " + name(c));
                }
            }
            auto variant_json = [&](const Variant & variant, const char * mode) {
                Json result{{"stats", options.stock_control ? Json(nullptr) : stats_json(variant.stats)},
                    {"execution_context", mode}, {"compute_wall_ms", variant.compute_wall_ms},
                    {"tensor_buffer_bytes", variant.tensor_buffer_bytes}, {"retained_device_delta_bytes", variant.retained_device_delta_bytes},
                    {"unused_f16_workspace_guard", {{"checked", variant.unused_workspace_guard_checked},
                        {"bytes", variant.unused_workspace_guard_bytes}}}};
                if (options.auto_control) result["workspace_probe"] = {{"checked", variant.workspace_probe_checked},
                    {"bytes", variant.workspace_probe_bytes}, {"expected_state", variant.workspace_probe_expected},
                    {"observed_state", variant.workspace_probe_observed}, {"changed_bytes", variant.workspace_probe_changed_bytes}};
                return result;
            };
            Json row{{"name", name(c)}, {"expected_fused", options.stock_control ? Json(nullptr) : Json(c.expected_fused)},
                {"candidate_geometry_supported", c.expected_fused}, {"output_bits_equal", true},
                {"input_fnv1a64", input_fingerprints}, {"all_outputs_finite", true}, {"all_inputs_and_guard_bytes_preserved", true}, {"output_floats", baseline.output.size()},
                {"output_fnv1a64", fingerprint(baseline.output.data(), baseline.output.size() * sizeof(float))},
                {"estimated_tensor_workspace_bytes", estimate_device_bytes(c)}};
            if (options.stock_control) {
                row["stock_env_off"] = variant_json(baseline, "pristine stock backend, environment=0");
                row["stock_env_on"] = variant_json(fused, "pristine stock backend, environment=1 (ignored)");
                row["fused_path_proven"] = false;
            } else {
                row["baseline"] = variant_json(baseline, "experimental backend, environment=0, stock path");
                row["candidate"] = variant_json(fused, "experimental backend, environment=1");
            }
            if (automatic) {
                // OFF's workspace witness must change; it is not a preserved input/output guard.
                row.erase("all_inputs_and_guard_bytes_preserved");
                row["all_input_bytes_and_input_guard_bytes_preserved"] = true;
                row["all_workspace_probe_expectations_met"] = true;
                row["auto"] = variant_json(*automatic, "environment unset, fresh default AUTO context after explicit diagnostic twins");
                row["auto"].update({{"zero_stats", all_stats_zero(automatic->stats)}, {"output_bits_equal", true},
                    {"all_outputs_finite", true}, {"all_inputs_and_guard_bytes_preserved", true},
                    {"output_floats", automatic->output.size()},
                    {"output_fnv1a64", fingerprint(automatic->output.data(), automatic->output.size() * sizeof(float))}});
            }
            report["results"].push_back(std::move(row));
            report.erase("pending_case"); save(options, report);
        }
        report["status"] = "passed"; report["passed_cases"] = selected.size();
        report["timing_scope"] = "synchronized actual-operator host wall time with CUDA graph replay disabled; not full-engine throughput";
        save(options, report); std::cout << report.dump(2) << '\n'; ggml_quantize_free(); return 0;
    } catch (const std::exception & error) {
        report["status"] = "failed"; report["failure"] = error.what();
        try { save(options, report); } catch (...) {}
        std::cerr << report.dump(2) << '\n'; return 1;
    }
}
