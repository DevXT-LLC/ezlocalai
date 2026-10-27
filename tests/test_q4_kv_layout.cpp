#include "fattn-q4-kv-policy.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

static_assert(std::is_standard_layout<ggml_cuda_fattn_q4_kv_stats>::value, "diagnostic stats ABI layout");
static_assert(sizeof(ggml_cuda_fattn_q4_kv_stats) == 17 * sizeof(uint64_t), "diagnostic stats ABI size");

namespace {
size_t checks = 0;
void check(bool condition, const std::string & message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

// Construct known packed layouts, independently of the policy's span helpers.
// Addresses are non-dereferenced metadata, spaced apart to cover long contexts.
ggml_cuda_q4_kv_policy valid(int queries = 4, int64_t context = 512,
        bool token_major_q = true, bool token_major_kv = true) {
    ggml_cuda_q4_kv_policy p;
    p.enabled = true;
    p.cc = 890;
    p.mask_present = true;
    const uint64_t q_row = 256 * sizeof(float), kv_row = 8 * 18;
    p.q.type = GGML_TYPE_F32;
    p.q.ne[0] = 256; p.q.ne[1] = queries; p.q.ne[2] = 24; p.q.ne[3] = 1;
    p.q.nb[0] = sizeof(float);
    p.q.nb[1] = q_row * (token_major_q ? 24 : 1);
    p.q.nb[2] = q_row * (token_major_q ? 1 : queries);
    p.q.nb[3] = q_row * queries * 24;
    p.q.data = uint64_t(1) << 40;
    p.q.bytes = q_row * queries * 24;
    for (auto * tensor : {&p.k, &p.v}) {
        tensor->type = GGML_TYPE_Q4_0;
        tensor->ne[0] = 256; tensor->ne[1] = context; tensor->ne[2] = 4; tensor->ne[3] = 1;
        tensor->nb[0] = 18;
        tensor->nb[1] = kv_row * (token_major_kv ? 4 : 1);
        tensor->nb[2] = kv_row * (token_major_kv ? 1 : context);
        // A single-sequence active view retains the full allocated cache stride.
        tensor->nb[3] = kv_row * 4 * uint64_t(context < 230144 ? 230144 : context);
        tensor->bytes = kv_row * 4 * uint64_t(context);
    }
    p.k.data = uint64_t(1) << 44;
    p.v.data = uint64_t(1) << 48;
    p.mask.type = GGML_TYPE_F16;
    p.mask.ne[0] = context; p.mask.ne[1] = 32; p.mask.ne[2] = p.mask.ne[3] = 1;
    p.mask.nb[0] = 2; p.mask.nb[1] = uint64_t(context) * 2;
    p.mask.nb[2] = p.mask.nb[3] = uint64_t(context) * 2 * 32;
    p.mask.bytes = uint64_t(context) * 2 * 32;
    p.mask.data = uint64_t(1) << 52;
    return p;
}

template<typename Mutate> void rejected(const std::string & name, Mutate mutate) {
    auto p = valid(); mutate(p);
    check(!ggml_cuda_q4_kv_eligible(p), "unsupported metadata accepted: " + name);
}

void environment_gate() {
    check(ggml_cuda_q4_kv_enabled(nullptr), "unset flag must permit guarded automatic activation");
    check(!ggml_cuda_q4_kv_diagnostics(nullptr), "automatic mode must not collect diagnostics");
    check(!ggml_cuda_q4_kv_enabled("0"), "explicit zero must disable fusion");
    check(ggml_cuda_q4_kv_diagnostics("0"), "explicit zero must provide stock-path diagnostics");
    for (const char * value : {"", "true", "yes", "01", "10", " 1", "1 ", "1\n", "-1"}) {
        check(!ggml_cuda_q4_kv_enabled(value), std::string("non-exact enable flag accepted: ") + value);
        check(!ggml_cuda_q4_kv_diagnostics(value), std::string("invalid flag enabled diagnostics: ") + value);
    }
    check(ggml_cuda_q4_kv_enabled("1"), "exact enable flag rejected");
    check(ggml_cuda_q4_kv_diagnostics("1"), "explicit one must provide fused-path diagnostics");

    const char * modes[] = {nullptr, "0", "1", "", "invalid"};
    for (const char * mode : modes) {
        auto p = valid();
        p.enabled = ggml_cuda_q4_kv_enabled(mode);
        check(ggml_cuda_q4_kv_eligible(p) == (mode == nullptr || std::string(mode) == "1"),
                "mode parser and guarded eligibility disagree");
        for (int cc : {750, 860, 870, 900, 1200, 1210}) {
            p.cc = cc;
            check(!ggml_cuda_q4_kv_eligible(p), "automatic or explicit mode broadened architecture eligibility");
        }
    }
}

void accepted_geometry() {
    for (int q : {3, 4}) {
        for (int64_t context : {256LL, 512LL, 1024LL, 8192LL, 130048LL, 130304LL, 230144LL}) {
            for (bool q_layout : {false, true}) {
                for (bool kv_layout : {false, true}) {
                    auto p = valid(q, context, q_layout, kv_layout);
                    check(ggml_cuda_q4_kv_eligible(p), "supported packed geometry rejected");
                    // Offsets remain aligned while each view's independent byte
                    // budget still covers its own logical span.
                    p.q.data += 16; p.k.data += 144; p.v.data += 576; p.mask.data += 16;
                    check(ggml_cuda_q4_kv_eligible(p), "supported aligned view rejected");
                    p.k.nb[3] = p.k.bytes; p.v.nb[3] = p.v.bytes;
                    check(ggml_cuda_q4_kv_eligible(p), "dense sequence stride rejected");
                }
            }
        }
    }
    auto p = valid();
    p.mask.ne[1] = 4; p.mask.bytes = p.mask.nb[2] = p.mask.nb[3] = 512 * 2 * 4;
    check(ggml_cuda_q4_kv_eligible(p), "unpadded but sufficient mask rows rejected");
    auto adjacent = valid();
    adjacent.v.data = adjacent.k.data + adjacent.k.bytes;
    check(ggml_cuda_q4_kv_eligible(adjacent), "adjacent half-open K/V ranges rejected");
    adjacent = valid();
    adjacent.mask.data = adjacent.q.data - adjacent.mask.bytes;
    check(ggml_cuda_q4_kv_eligible(adjacent), "adjacent half-open mask/Q ranges rejected");
    p.max_bias = -0.0f; p.logit_softcap = -0.0f;
    check(ggml_cuda_q4_kv_eligible(p), "negative zero feature parameters rejected");
}

void excluded_geometry() {
    rejected("explicitly disabled policy", [](auto & p) { p.enabled = false; });
    for (int cc : {0, 750, 800, 860, 870, 889, 891, 900, 1200, 1210})
        rejected("unqualified architecture", [=](auto & p) { p.cc = cc; });
    rejected("missing mask", [](auto & p) { p.mask_present = false; });
    rejected("attention sinks", [](auto & p) { p.has_sinks = true; });
    rejected("sparse gather", [](auto & p) { p.use_sparse = true; });
    rejected("shared KV view", [](auto & p) { p.shared_kv = true; });
    rejected("ALiBi", [](auto & p) { p.max_bias = 1.0f; });
    rejected("softcap", [](auto & p) { p.logit_softcap = 1.0f; });
    rejected("NaN bias", [](auto & p) { p.max_bias = std::numeric_limits<float>::quiet_NaN(); });
    rejected("NaN softcap", [](auto & p) { p.logit_softcap = std::numeric_limits<float>::quiet_NaN(); });
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_COUNT}) {
        rejected("non-Q4 K", [=](auto & p) { p.k.type = type; });
        rejected("non-Q4 V", [=](auto & p) { p.v.type = type; });
    }
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_COUNT})
        rejected("non-F32 Q", [=](auto & p) { p.q.type = type; });
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_COUNT})
        rejected("non-F16 mask", [=](auto & p) { p.mask.type = type; });
    for (int64_t rows : {0LL, 1LL, 2LL, 5LL, 8LL, -1LL})
        rejected("unsupported Q rows", [=](auto & p) { p.q.ne[1] = rows; });
    rejected("Q channel dimension", [](auto & p) { p.q.ne[0] = 128; });
    rejected("KV channel dimension", [](auto & p) { p.k.ne[0] = p.v.ne[0] = 128; });
    rejected("Q heads GQA4", [](auto & p) { p.q.ne[2] = 16; });
    rejected("Q heads GQA8", [](auto & p) { p.q.ne[2] = 32; });
    rejected("KV heads", [](auto & p) { p.k.ne[2] = p.v.ne[2] = 2; });
    rejected("multiple Q sequences", [](auto & p) { p.q.ne[3] = 2; });
    rejected("multiple KV sequences", [](auto & p) { p.k.ne[3] = p.v.ne[3] = 2; });
    for (int64_t context : {-256LL, 0LL, 1LL, 255LL, 257LL, 511LL, 513LL})
        rejected("unpadded KV context", [=](auto & p) { p.k.ne[1] = p.v.ne[1] = context; });
    rejected("unequal KV shapes", [](auto & p) { p.v.ne[1] += 256; });
    rejected("mixed KV physical layouts", [](auto & p) {
        p.v.nb[1] = 144; p.v.nb[2] = p.v.ne[1] * 144;
    });
    rejected("mask context", [](auto & p) { --p.mask.ne[0]; });
    rejected("too few mask queries", [](auto & p) { p.mask.ne[1] = 3; });
    rejected("mask head dimensions", [](auto & p) { p.mask.ne[2] = 2; });
    rejected("mask multiple sequences", [](auto & p) { p.mask.ne[3] = 2; });
    rejected("mask non-dense row", [](auto & p) { p.mask.nb[1] += 16; });
    rejected("mask non-dense channel", [](auto & p) { p.mask.nb[0] = 4; });
    rejected("mask short plane", [](auto & p) { p.mask.nb[2] = 16; });
    rejected("mask short sequence", [](auto & p) { p.mask.nb[3] = 16; });
    rejected("Q non-dense channel", [](auto & p) { p.q.nb[0] = 8; });
    rejected("KV invalid block stride", [](auto & p) { p.k.nb[0] = p.v.nb[0] = 20; });
    rejected("KV row gap", [](auto & p) { p.k.nb[1] += 16; p.v.nb[1] += 16; });
    rejected("KV head gap", [](auto & p) { p.k.nb[2] += 16; p.v.nb[2] += 16; });
    rejected("Q row gap", [](auto & p) { p.q.nb[1] += 16; });
    rejected("Q head gap", [](auto & p) { p.q.nb[2] += 16; });
    rejected("Q sequence narrowing", [](auto & p) { p.q.nb[3] = uint64_t(INT32_MAX) + 1; });
    rejected("Q too short sequence", [](auto & p) { p.q.nb[3] = p.q.bytes - 16; });
    rejected("KV too short sequence", [](auto & p) { p.k.nb[3] = p.k.bytes - 4; });
    rejected("KV row narrowing", [](auto & p) { p.k.nb[1] = p.v.nb[1] = uint64_t(INT32_MAX) + 1; });
    rejected("KV sequence narrowing", [](auto & p) { p.k.nb[3] = p.v.nb[3] = uint64_t(INT64_MAX) + 1; });
    for (int tensor = 0; tensor < 4; ++tensor) {
        rejected("missing data", [=](auto & p) {
            auto * t = tensor == 0 ? &p.q : tensor == 1 ? &p.k : tensor == 2 ? &p.v : &p.mask;
            t->data = 0;
        });
        rejected("unaligned data", [=](auto & p) {
            auto * t = tensor == 0 ? &p.q : tensor == 1 ? &p.k : tensor == 2 ? &p.v : &p.mask;
            t->data += 2;
        });
        rejected("short backing span", [=](auto & p) {
            auto * t = tensor == 0 ? &p.q : tensor == 1 ? &p.k : tensor == 2 ? &p.v : &p.mask;
            --t->bytes;
        });
        rejected("address overflow", [=](auto & p) {
            auto * t = tensor == 0 ? &p.q : tensor == 1 ? &p.k : tensor == 2 ? &p.v : &p.mask;
            t->data = UINT64_MAX - 15;
        });
        for (int dim = 0; dim < 4; ++dim) {
            rejected("zero dimension", [=](auto & p) {
                auto * t = tensor == 0 ? &p.q : tensor == 1 ? &p.k : tensor == 2 ? &p.v : &p.mask;
                t->ne[dim] = 0;
            });
            rejected("negative dimension", [=](auto & p) {
                auto * t = tensor == 0 ? &p.q : tensor == 1 ? &p.k : tensor == 2 ? &p.v : &p.mask;
                t->ne[dim] = -1;
            });
            rejected("dimension narrowing", [=](auto & p) {
                auto * t = tensor == 0 ? &p.q : tensor == 1 ? &p.k : tensor == 2 ? &p.v : &p.mask;
                t->ne[dim] = int64_t(INT32_MAX) + 1;
            });
            rejected("zero stride", [=](auto & p) {
                auto * t = tensor == 0 ? &p.q : tensor == 1 ? &p.k : tensor == 2 ? &p.v : &p.mask;
                t->nb[dim] = 0;
            });
        }
    }
    rejected("KV same address", [](auto & p) { p.v.data = p.k.data; });
    rejected("KV partial overlap", [](auto & p) { p.v.data = p.k.data + 144; });
    rejected("Q/K overlap", [](auto & p) { p.q.data = p.k.data + 16; });
    rejected("Q/V overlap", [](auto & p) { p.q.data = p.v.data + 16; });
    rejected("mask/K overlap", [](auto & p) { p.mask.data = p.k.data + 16; });
    rejected("mask/V overlap", [](auto & p) { p.mask.data = p.v.data + 16; });
    rejected("mask/Q overlap", [](auto & p) { p.mask.data = p.q.data + 16; });
}
}

int main() {
    try {
        environment_gate();
        accepted_geometry();
        excluded_geometry();
        std::cout << "{\"status\":\"passed\",\"suite\":\"q4_kv_layout\",\"gpu_initialized\":false,\"checks\":"
                  << checks << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "q4_kv_layout failed: " << error.what() << '\n';
        return 1;
    }
}
