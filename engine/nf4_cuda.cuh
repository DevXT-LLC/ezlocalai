#pragma once
#include "nf4_ggml.hpp"

namespace ezlocalai {
static_assert(sizeof(Nf4CustomParams) == sizeof(ggml_custom_op_params));
static_assert(offsetof(Nf4CustomParams, userdata) == offsetof(ggml_custom_op_params, userdata));
// Use the same exact tables as the source-validated CPU quantizer.
static __device__ __constant__ std::array<float, 256> nf4_cuda_dynamic = nf4_detail::dynamic_code;
constexpr std::array<float, 16> nf4_cuda_values() {
    std::array<float, 16> values{};
    for (size_t i = 0; i < values.size(); ++i) values[i] = nf4_detail::nf4_dequantization_lut[i];
    return values;
}
static __device__ __constant__ std::array<float, 16> nf4_cuda_table = nf4_cuda_values();

static __device__ __forceinline__ float nf4_cuda_float(const uint8_t *source) {
    // A contiguous I8 view may start at an unaligned byte offset.
    const uint32_t bits = uint32_t(source[0]) | (uint32_t(source[1]) << 8)
        | (uint32_t(source[2]) << 16) | (uint32_t(source[3]) << 24);
    return __uint_as_float(bits);
}

static __global__ void nf4_cuda_expand_kernel(const uint8_t *source, uint16_t *destination,
                                             Nf4StorageLayout layout, bool bf16) {
    const auto *dynamic = reinterpret_cast<const float *>(&nf4_cuda_dynamic);
    const auto *table = reinterpret_cast<const float *>(&nf4_cuda_table);
    const float offset = nf4_cuda_float(source + layout.offset);
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < layout.count;
         i += size_t(blockDim.x) * gridDim.x) {
        const size_t block = i / 64;
        const float maximum = nf4_cuda_float(source + layout.maxima + (block / 256) * sizeof(float));
        const float scale = __fadd_rn(__fmul_rn(dynamic[source[layout.statistics + block]], maximum), offset);
        const auto code = (source[i / 2] >> (i % 2 ? 0 : 4)) & 15;
        const float value = __fmul_rn(table[code], scale);
        if (bf16) {
            const uint32_t bits = __float_as_uint(value);
            destination[i] = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
        } else destination[i] = __half_as_ushort(__float2half_rn(value));
    }
}

static void nf4_cuda_expand(ggml_backend_cuda_context &context, ggml_tensor *destination) {
    GGML_ASSERT(is_nf4_ggml(destination));
    const Nf4StorageLayout layout(ggml_nelements(destination));
    const unsigned blocks = static_cast<unsigned>(std::min(size_t(65535), (layout.count + 255) / 256));
    nf4_cuda_expand_kernel<<<blocks, 256, 0, context.stream()>>>(
        static_cast<const uint8_t *>(destination->src[0]->data), static_cast<uint16_t *>(destination->data),
        layout, destination->type == GGML_TYPE_BF16);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ezlocalai
