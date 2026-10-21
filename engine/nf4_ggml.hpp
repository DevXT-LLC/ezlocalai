#pragma once
#include "ezlocalai/nf4_storage.hpp"
#include "ggml.h"

namespace ezlocalai {
// The private CUDA extension recognizes this custom operation. Its CPU callback
// remains a normal GGML custom op, including when a scheduler selects CPU.
inline constexpr uintptr_t nf4_ggml_tag = 0x455A4E463401ULL;
struct Nf4CustomParams { ggml_custom_op_t fun; int n_tasks; void *userdata; };
static_assert(sizeof(Nf4CustomParams) <= GGML_MAX_OP_PARAMS);

inline bool is_nf4_ggml(const ggml_tensor *tensor) {
    if (!tensor || tensor->op != GGML_OP_CUSTOM) return false;
    Nf4CustomParams parameters;
    std::memcpy(&parameters, tensor->op_params, sizeof(parameters));
    if (reinterpret_cast<uintptr_t>(parameters.userdata) != nf4_ggml_tag) return false;
    if ((tensor->type != GGML_TYPE_BF16 && tensor->type != GGML_TYPE_F16)
        || !ggml_is_contiguous(tensor) || !tensor->src[0] || tensor->src[1]
        || tensor->src[0]->type != GGML_TYPE_I8 || !ggml_is_contiguous(tensor->src[0])) return false;
    try { return ggml_nbytes(tensor->src[0]) == Nf4StorageLayout(ggml_nelements(tensor)).bytes; }
    catch (const std::invalid_argument &) { return false; }
}

inline void nf4_ggml_cpu(ggml_tensor *destination, int thread, int threads, void *) {
    GGML_ASSERT(is_nf4_ggml(destination));
    const Nf4StorageLayout layout(ggml_nelements(destination));
    const auto *source = static_cast<const uint8_t *>(destination->src[0]->data);
    auto *output = static_cast<uint16_t *>(destination->data);
    const size_t chunk = layout.count / threads + (layout.count % threads != 0);
    const size_t begin = std::min(layout.count, chunk * thread), end = begin + std::min(chunk, layout.count - begin);
    for (size_t i = begin; i < end; ++i) {
        const float value = nf4_storage_value(source, layout, i);
        output[i] = destination->type == GGML_TYPE_BF16 ? nf4_detail::to_bf16(value) : nf4_detail::to_f16(value);
    }
}

inline ggml_tensor *nf4_ggml_expand(ggml_context *context, ggml_tensor *storage,
                                  int64_t columns, int64_t rows, ggml_type type) {
    if (columns <= 0 || rows <= 0 || columns > INT64_MAX / rows
        || (type != GGML_TYPE_BF16 && type != GGML_TYPE_F16))
        throw std::invalid_argument("Invalid NF4 matrix shape or type");
    const Nf4StorageLayout layout(static_cast<size_t>(columns * rows));
    if (!storage || storage->type != GGML_TYPE_I8 || !ggml_is_contiguous(storage)
        || ggml_nbytes(storage) != layout.bytes) throw std::invalid_argument("Invalid NF4 storage tensor");
    return ggml_custom_4d(context, type, columns, rows, 1, 1, &storage, 1, nf4_ggml_cpu,
        GGML_N_TASKS_MAX, reinterpret_cast<void *>(nf4_ggml_tag));
}
} // namespace ezlocalai
