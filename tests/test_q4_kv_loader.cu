// Physical-GPU oracle. Default invocation does not initialize CUDA.
// Compare the actual managed tile loader with the stock converter and loader.
#include "convert.cuh"
#include "fattn-mma-f16.cuh"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::json;
namespace {
constexpr size_t guard_bytes = 128;
constexpr unsigned char guard_value = 0xa5;
constexpr int channels = 256;
constexpr int head_count = 4;
constexpr int tile_rows = 32;
constexpr int stride_tile = channels / 2;
constexpr size_t blocks_per_row = channels / QK4_0;
constexpr size_t packed_row_bytes = blocks_per_row * sizeof(block_q4_0);
static_assert(sizeof(block_q4_0) == 18 && QK4_0 == 32, "pinned Q4 format changed");
static_assert(packed_row_bytes == 144, "pinned row shape changed");

void check(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}
void cuda_check(cudaError_t value, const char * operation) {
    if (value != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(value));
}
struct DeviceBytes {
    unsigned char * pointer = nullptr;
    size_t size = 0;
    explicit DeviceBytes(size_t bytes) : size(bytes) {
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&pointer), bytes), "cudaMalloc");
        cuda_check(cudaMemset(pointer, guard_value, bytes), "cudaMemset guards");
    }
    ~DeviceBytes() { if (pointer) cudaFree(pointer); }
    DeviceBytes(const DeviceBytes &) = delete;
    DeviceBytes & operator=(const DeviceBytes &) = delete;
    std::vector<unsigned char> read() const {
        std::vector<unsigned char> result(size);
        cuda_check(cudaMemcpy(result.data(), pointer, size, cudaMemcpyDeviceToHost), "read device buffer");
        return result;
    }
};

__global__ void decode_all_chunks(const block_q4_0 * source, half * destination, int block_count) {
    const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= block_count * 4) return;
    const int block = chunk / 4;
    ggml_cuda_fattn_q4_0_decode_8(source + block, (chunk % 4) * 8, destination + chunk * 8);
}

// Both specializations execute the production load_tile implementation. The
// reference specialization consumes the independently converted F16 allocation.
// Shared-memory guard bands catch either writes or swizzles outside the tile.
template<bool packed_q4, bool async_load>
__global__ void capture_production_tile(const unsigned char * source, half * destination,
        unsigned int * shared_guard_errors, int rows, int token_stride_h2, int head_stride_bytes,
        int first_row) {
    constexpr size_t tile_bytes = tile_rows * channels * sizeof(half);
    __shared__ __align__(16) unsigned char storage[guard_bytes + tile_bytes + guard_bytes];
    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    for (size_t i = thread; i < sizeof(storage); i += blockDim.x * blockDim.y) storage[i] = guard_value;
    __syncthreads();
    auto * tile = reinterpret_cast<half2 *>(storage + guard_bytes);
    const auto * head = reinterpret_cast<const half2 *>(source + int64_t(blockIdx.y) * head_stride_bytes);
    const int row = first_row + blockIdx.x * tile_rows;
    flash_attn_ext_f16_load_tile<stride_tile, 4, tile_rows, async_load, false, false, packed_q4>(
            head, tile, channels / 2, token_stride_h2, row, rows - row, nullptr);
    if constexpr (async_load) cp_async_wait_all();
    __syncthreads();
    for (size_t i = thread; i < guard_bytes; i += blockDim.x * blockDim.y) {
        if (storage[i] != guard_value || storage[guard_bytes + tile_bytes + i] != guard_value) {
            atomicAdd(shared_guard_errors, 1u);
        }
    }
    const size_t tile_index = size_t(blockIdx.y) * gridDim.x + blockIdx.x;
    auto * output = reinterpret_cast<unsigned char *>(destination) + tile_index * tile_bytes;
    for (size_t i = thread; i < tile_bytes; i += blockDim.x * blockDim.y) output[i] = storage[guard_bytes + i];
}

bool is_half_nan(uint16_t bits) { return (bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0; }
struct Comparison {
    size_t values = 0;
    size_t nan_values = 0;
    size_t nan_payload_differences = 0;
    size_t bit_mismatches = 0;
    size_t classification_mismatches = 0;
};
Comparison compare_half_bits(const std::vector<unsigned char> & actual,
        const std::vector<unsigned char> & expected, size_t offset, size_t values) {
    check(actual.size() == expected.size() && offset + values * 2 <= actual.size(), "comparison bounds");
    Comparison result;
    result.values = values;
    for (size_t i = 0; i < values; ++i) {
        uint16_t a, e;
        std::memcpy(&a, actual.data() + offset + i * 2, sizeof(a));
        std::memcpy(&e, expected.data() + offset + i * 2, sizeof(e));
        if (is_half_nan(a) || is_half_nan(e)) {
            if (!(is_half_nan(a) && is_half_nan(e))) ++result.classification_mismatches;
            else {
                ++result.nan_values;
                if (a != e) ++result.nan_payload_differences;
            }
        } else if (a != e) ++result.bit_mismatches;
    }
    check(result.bit_mismatches == 0, "non-NaN F16 operand bits differ from stock conversion/loading");
    check(result.classification_mismatches == 0, "NaN classification differs from stock conversion/loading");
    return result;
}
void verify_guards(const std::vector<unsigned char> & bytes, size_t begin, size_t used) {
    check(begin + used <= bytes.size(), "guard bounds");
    for (size_t i = 0; i < begin; ++i) check(bytes[i] == guard_value, "prefix guard overwritten");
    for (size_t i = begin + used; i < bytes.size(); ++i) check(bytes[i] == guard_value, "suffix guard overwritten");
}
Json comparison_json(const Comparison & value) {
    return {{"compared_half_operands", value.values}, {"non_nan_bit_mismatches", value.bit_mismatches},
            {"nan_classification_mismatches", value.classification_mismatches},
            {"both_nan_operands", value.nan_values}, {"nan_payload_differences", value.nan_payload_differences}};
}

std::vector<unsigned char> make_input(int block_count, unsigned pattern) {
    std::vector<unsigned char> result(size_t(block_count) * sizeof(block_q4_0));
    for (int block = 0; block < block_count; ++block) {
        const uint16_t scale = uint16_t(block); // Includes every scale bit pattern in exhaustive mode.
        std::memcpy(result.data() + size_t(block) * sizeof(block_q4_0), &scale, sizeof(scale));
        for (int nibble = 0; nibble < 16; ++nibble) {
            const unsigned low = (unsigned(nibble) + pattern) & 15u;
            const unsigned high = (15u - unsigned(nibble) + pattern * 3u) & 15u;
            result[size_t(block) * sizeof(block_q4_0) + 2 + nibble] = low | (high << 4);
        }
    }
    return result;
}

Json run_layout(int block_count, bool token_major, size_t input_skew, unsigned pattern) {
    check(block_count % int(head_count * blocks_per_row * tile_rows) == 0, "complete tile geometry required");
    const int rows = block_count / (head_count * blocks_per_row);
    const size_t elements = size_t(block_count) * QK4_0;
    const size_t half_bytes = elements * sizeof(half);
    const size_t input_offset = guard_bytes + input_skew;
    auto source = make_input(block_count, pattern);
    DeviceBytes input(input_offset + source.size() + guard_bytes);
    DeviceBytes converted(guard_bytes + half_bytes + guard_bytes);
    DeviceBytes primitive(guard_bytes + half_bytes + guard_bytes);
    DeviceBytes reference_tile(guard_bytes + half_bytes + guard_bytes);
    DeviceBytes fused_tile(guard_bytes + half_bytes + guard_bytes);
    DeviceBytes guard_errors(sizeof(unsigned int));
    cuda_check(cudaMemset(guard_errors.pointer, 0, sizeof(unsigned int)), "clear shared guard errors");
    cuda_check(cudaMemcpy(input.pointer + input_offset, source.data(), source.size(), cudaMemcpyHostToDevice), "upload Q4 blocks");
    const auto * packed = input.pointer + input_offset;
    auto * output_half = reinterpret_cast<half *>(converted.pointer + guard_bytes);
    const auto converter = ggml_get_to_fp16_cuda(GGML_TYPE_Q4_0);
    check(converter != nullptr, "original Q4_0-to-F16 converter is unavailable");
    converter(packed, output_half, elements, nullptr);
    cuda_check(cudaGetLastError(), "original converter launch");
    decode_all_chunks<<<(block_count * 4 + 127) / 128, 128>>>(
            reinterpret_cast<const block_q4_0 *>(packed), reinterpret_cast<half *>(primitive.pointer + guard_bytes), block_count);
    cuda_check(cudaGetLastError(), "production decode primitive launch");
    cuda_check(cudaDeviceSynchronize(), "primitive oracle synchronize");
    auto converted_host = converted.read();
    auto primitive_host = primitive.read();
    auto primitive_comparison = compare_half_bits(primitive_host, converted_host, guard_bytes, elements);
    verify_guards(converted_host, guard_bytes, half_bytes);
    verify_guards(primitive_host, guard_bytes, half_bytes);

    const dim3 grid(rows / tile_rows, head_count);
    const dim3 threads(32, 4);
    const int raw_token_stride = int(packed_row_bytes) * (token_major ? head_count : 1);
    const int raw_head_stride = int(packed_row_bytes) * (token_major ? 1 : rows);
    const int f16_token_stride = channels * sizeof(half) * (token_major ? head_count : 1);
    const int f16_head_stride = channels * sizeof(half) * (token_major ? 1 : rows);
    Json tile_rows_json = Json::array();
    // Exercise the asynchronous policy used by the production two-stage MMA,
    // and the synchronous policy to detect drift in either shared-memory path.
    for (const bool async_load : {true, false}) {
        cuda_check(cudaMemset(reference_tile.pointer, guard_value, reference_tile.size), "reset reference guards");
        cuda_check(cudaMemset(fused_tile.pointer, guard_value, fused_tile.size), "reset candidate guards");
        auto * ref = reinterpret_cast<half *>(reference_tile.pointer + guard_bytes);
        auto * fused = reinterpret_cast<half *>(fused_tile.pointer + guard_bytes);
        auto * errors = reinterpret_cast<unsigned int *>(guard_errors.pointer);
        if (async_load) {
            capture_production_tile<false, true><<<grid, threads>>>(converted.pointer + guard_bytes, ref, errors,
                    rows, f16_token_stride / sizeof(half2), f16_head_stride, 0);
            capture_production_tile<true, true><<<grid, threads>>>(packed, fused, errors,
                    rows, raw_token_stride / sizeof(half2), raw_head_stride, 0);
        } else {
            capture_production_tile<false, false><<<grid, threads>>>(converted.pointer + guard_bytes, ref, errors,
                    rows, f16_token_stride / sizeof(half2), f16_head_stride, 0);
            capture_production_tile<true, false><<<grid, threads>>>(packed, fused, errors,
                    rows, raw_token_stride / sizeof(half2), raw_head_stride, 0);
        }
        cuda_check(cudaGetLastError(), "actual tile loader launch");
        cuda_check(cudaDeviceSynchronize(), "tile oracle synchronize");
        auto reference_host = reference_tile.read();
        auto actual_host = fused_tile.read();
        const auto compared = compare_half_bits(actual_host, reference_host, guard_bytes, elements);
        verify_guards(reference_host, guard_bytes, half_bytes);
        verify_guards(actual_host, guard_bytes, half_bytes);
        unsigned errors_host = 0;
        cuda_check(cudaMemcpy(&errors_host, guard_errors.pointer, sizeof(errors_host), cudaMemcpyDeviceToHost), "read shared guards");
        check(errors_host == 0, "actual tile loader modified shared-memory guard bands");
        auto row = comparison_json(compared);
        row["async_policy"] = async_load;
        row["shared_guard_errors"] = errors_host;
        tile_rows_json.push_back(row);
    }
    const auto preserved = input.read();
    verify_guards(preserved, input_offset, source.size());
    check(std::memcmp(preserved.data() + input_offset, source.data(), source.size()) == 0, "loader/converter modified packed source");
    return {{"layout", token_major ? "production_token_major" : "head_major"},
            {"logical_rows_per_head", rows}, {"head_count", head_count}, {"channel_count", channels},
            {"packed_token_stride_bytes", raw_token_stride}, {"packed_head_stride_bytes", raw_head_stride},
            {"packed_input_skew_bytes", input_skew}, {"pattern", pattern},
            {"production_decode_primitive", comparison_json(primitive_comparison)},
            {"production_tile_loader", tile_rows_json}, {"packed_source_preserved", true},
            {"global_guard_bands_preserved", true}};
}

Json run_long_offset(int rows, bool token_major) {
    check(rows >= tile_rows && rows % 256 == 0, "long-offset context padding");
    const int blocks = rows * head_count * blocks_per_row;
    const size_t elements = size_t(blocks) * QK4_0;
    const size_t converted_bytes = elements * sizeof(half);
    const size_t tile_elements = size_t(tile_rows) * head_count * channels;
    const size_t tile_bytes = tile_elements * sizeof(half);
    const size_t input_offset = guard_bytes + 16;
    auto source = make_input(blocks, 3);
    DeviceBytes input(input_offset + source.size() + guard_bytes);
    DeviceBytes converted(guard_bytes + converted_bytes + guard_bytes);
    DeviceBytes reference_tile(guard_bytes + tile_bytes + guard_bytes);
    DeviceBytes fused_tile(guard_bytes + tile_bytes + guard_bytes);
    DeviceBytes guard_errors(sizeof(unsigned int));
    cuda_check(cudaMemset(guard_errors.pointer, 0, sizeof(unsigned int)), "clear long-offset shared guards");
    cuda_check(cudaMemcpy(input.pointer + input_offset, source.data(), source.size(), cudaMemcpyHostToDevice), "upload long-offset source");
    ggml_get_to_fp16_cuda(GGML_TYPE_Q4_0)(input.pointer + input_offset,
            reinterpret_cast<half *>(converted.pointer + guard_bytes), elements, nullptr);
    cuda_check(cudaGetLastError(), "long-offset stock converter launch");
    const int first_row = rows - tile_rows;
    const int raw_token_stride = int(packed_row_bytes) * (token_major ? head_count : 1);
    const int raw_head_stride = int(packed_row_bytes) * (token_major ? 1 : rows);
    const int f16_token_stride = channels * sizeof(half) * (token_major ? head_count : 1);
    const int f16_head_stride = channels * sizeof(half) * (token_major ? 1 : rows);
    const dim3 grid(1, head_count), threads(32, 4);
    auto * errors = reinterpret_cast<unsigned int *>(guard_errors.pointer);
    capture_production_tile<false, true><<<grid, threads>>>(converted.pointer + guard_bytes,
            reinterpret_cast<half *>(reference_tile.pointer + guard_bytes), errors, rows,
            f16_token_stride / sizeof(half2), f16_head_stride, first_row);
    cuda_check(cudaGetLastError(), "long-offset stock tile launch");
    capture_production_tile<true, true><<<grid, threads>>>(input.pointer + input_offset,
            reinterpret_cast<half *>(fused_tile.pointer + guard_bytes), errors, rows,
            raw_token_stride / sizeof(half2), raw_head_stride, first_row);
    cuda_check(cudaGetLastError(), "long-offset fused tile launch");
    cuda_check(cudaDeviceSynchronize(), "long-offset synchronize");
    const auto reference = reference_tile.read(), actual = fused_tile.read();
    const auto compared = compare_half_bits(actual, reference, guard_bytes, tile_elements);
    verify_guards(reference, guard_bytes, tile_bytes);
    verify_guards(actual, guard_bytes, tile_bytes);
    unsigned errors_host = 0;
    cuda_check(cudaMemcpy(&errors_host, guard_errors.pointer, sizeof(errors_host), cudaMemcpyDeviceToHost), "read long-offset shared guards");
    check(errors_host == 0, "long-offset loader modified shared-memory guards");
    const auto preserved = input.read();
    verify_guards(preserved, input_offset, source.size());
    check(std::memcmp(preserved.data() + input_offset, source.data(), source.size()) == 0, "long-offset packed source changed");
    auto row = comparison_json(compared);
    row["layout"] = token_major ? "production_token_major" : "head_major";
    row["context_rows"] = rows;
    row["first_tile_row"] = first_row;
    row["packed_token_stride_bytes"] = raw_token_stride;
    row["packed_head_stride_bytes"] = raw_head_stride;
    row["shared_guard_errors"] = errors_host;
    row["packed_source_preserved"] = true;
    row["device_allocation_bytes"] = input.size + converted.size + reference_tile.size + fused_tile.size + guard_errors.size;
    return row;
}
}

int main(int argc, char ** argv) {
    bool execute = false, quick = false;
    std::string output;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--execute-on-gpu0") execute = true;
            else if (arg == "--quick") quick = true;
            else if (arg == "--output" && i + 1 < argc) output = argv[++i];
            else if (arg == "--help") {
                std::cout << "test_q4_kv_loader [--execute-on-gpu0] [--quick] [--output PATH]\n";
                return 0;
            } else throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
        Json report{{"schema_version", 1}, {"suite", "q4_kv_loader"},
                    {"gpu_execution_requested", execute}, {"status", "not_run"},
                    {"nan_comparison_policy", "Both-NaN operands compare classification; every other operand compares all16bits, including signed zero and infinity"}};
        if (execute) {
            cudaDeviceProp property{};
            cuda_check(cudaGetDeviceProperties(&property, 0), "get GPU0 properties");
            check(property.major == 8 && property.minor == 9, "this physical oracle is restricted to SM89");
            cuda_check(cudaSetDevice(0), "select GPU0");
            report["device_sm"] = property.major * 100 + property.minor * 10;
            report["cases"] = Json::array();
            const int count = quick ? 1024 : 65536;
            for (const bool production_layout : {false, true}) {
                for (const size_t skew : {size_t(0), size_t(2)}) {
                    report["cases"].push_back(run_layout(count, production_layout, skew, 0));
                }
            }
            report["long_offset_cases"] = Json::array();
            if (!quick) {
                for (const int rows : {8192, 130048, 130304, 230144}) {
                    for (const bool token_major : {false, true}) {
                        report["long_offset_cases"].push_back(run_long_offset(rows, token_major));
                    }
                }
            }
            report["scale_bit_patterns"] = count;
            report["nibble_values_per_scale"] = 16;
            report["exact_numerical_path"] = "Original exported ggml_get_to_fp16_cuda(GGML_TYPE_Q4_0), production decode8 and production swizzled tile loader";
            report["status"] = "passed";
        }
        if (!output.empty()) {
            std::ofstream stream(output);
            check(bool(stream), "cannot open requested JSON output");
            stream << report.dump(2) << '\n';
            check(bool(stream), "cannot write requested JSON output");
        }
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << Json{{"status", "failed"}, {"suite", "q4_kv_loader"}, {"error", error.what()}}.dump() << '\n';
        return 1;
    }
}
