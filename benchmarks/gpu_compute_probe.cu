// Standalone diagnostic; deliberately separate from worker/CMake targets.
// Build (no GPU required):
//   nvcc -std=c++17 -O2 benchmarks/gpu_compute_probe.cu -lcublas -ldl -o gpu_compute_probe
// Offline plan: ./gpu_compute_probe --plan --max-bytes 805306368
// Run only in an operator-owned idle window, with external clocks/power capture:
//   ./gpu_compute_probe --run --device 0 --max-bytes 805306368
// CUDA 12.8 primary API reference, including TN integer GEMM requirements:
// https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasgemmex
// This measures explicit GEMM arithmetic, not model quality or unlock causality.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint64_t MiB = 1024ULL * 1024;
constexpr uint64_t workspace_bytes = 32 * MiB;
constexpr uint64_t library_headroom_bytes = 128 * MiB;
constexpr const char * schema = "ezlocalai_gpu_compute_probe_v1";

struct Options {
    bool run = false;
    bool plan = false;
    int device = -1;
    uint64_t max_bytes = 0;
    uint64_t min_free_bytes = 1024 * MiB;
    int warmups = 10;
    int samples = 7;
    int iterations = 20;
    std::vector<int> sizes{4096, 8192};
    std::vector<std::string> dtypes{"fp16", "bf16", "int8"};
};

std::string quoted(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}

uint64_t number(const std::string & value, const std::string & name) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error(name + " requires an unsigned decimal integer");
    }
    size_t used = 0;
    const auto result = std::stoull(value, &used);
    if (used != value.size()) throw std::runtime_error("invalid " + name);
    return result;
}

std::vector<std::string> split(const std::string & value) {
    std::vector<std::string> result;
    size_t begin = 0;
    do {
        const auto end = value.find(',', begin);
        const auto part = value.substr(begin, end == std::string::npos ? end : end - begin);
        if (part.empty()) throw std::runtime_error("empty comma-separated argument");
        result.push_back(part);
        if (end == std::string::npos) break;
        begin = end + 1;
    } while (true);
    return result;
}

Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--run") { o.run = true; continue; }
        if (arg == "--plan") { o.plan = true; continue; }
        if (arg == "--help") {
            std::cout << "Usage: gpu_compute_probe (--plan | --run --device INDEX) --max-bytes BYTES\n"
                "  [--sizes 4096,8192] [--dtypes fp16,bf16,int8]\n"
                "  [--warmups 10] [--samples 7] [--iterations 20]\n"
                "  [--min-free-bytes 1073741824]\n"
                "--plan and --help do not call CUDA. --run requires an explicit device and allocation cap.\n"
                "The cap covers explicit device buffers/workspace; context/library overhead is reported separately.\n"
                "Capture clocks, power, temperature and throttling externally. No OOM retries.\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--device") {
            const auto n = number(value, arg);
            if (n > 1024) throw std::runtime_error("device index exceeds 1024");
            o.device = int(n);
        } else if (arg == "--max-bytes") o.max_bytes = number(value, arg);
        else if (arg == "--min-free-bytes") o.min_free_bytes = number(value, arg);
        else if (arg == "--sizes") {
            o.sizes.clear();
            for (const auto & part : split(value)) {
                const auto n = number(part, arg);
                if (n != 4096 && n != 8192) throw std::runtime_error("sizes must be 4096 or 8192");
                if (std::find(o.sizes.begin(), o.sizes.end(), int(n)) != o.sizes.end()) throw std::runtime_error("duplicate size");
                o.sizes.push_back(int(n));
            }
        } else if (arg == "--dtypes") {
            o.dtypes.clear();
            for (const auto & part : split(value)) {
                if (part != "fp16" && part != "bf16" && part != "int8") throw std::runtime_error("unknown dtype: " + part);
                if (std::find(o.dtypes.begin(), o.dtypes.end(), part) != o.dtypes.end()) throw std::runtime_error("duplicate dtype");
                o.dtypes.push_back(part);
            }
        } else if (arg == "--warmups" || arg == "--samples" || arg == "--iterations") {
            const auto n = number(value, arg);
            const auto limit = arg == "--samples" ? 64ULL : 1000ULL;
            if (n == 0 || n > limit) throw std::runtime_error(arg + " outside allowed range");
            if (arg == "--warmups") o.warmups = int(n);
            else if (arg == "--samples") o.samples = int(n);
            else o.iterations = int(n);
        } else throw std::runtime_error("unknown option: " + arg);
    }
    if (o.run == o.plan) throw std::runtime_error("select exactly one of --plan and --run");
    if (o.run && o.device < 0) throw std::runtime_error("--run requires --device");
    if (o.max_bytes == 0) throw std::runtime_error("an explicit positive --max-bytes is required");
    if (o.min_free_bytes < 256 * MiB) throw std::runtime_error("--min-free-bytes must retain at least 256 MiB");
    if (o.max_bytes > std::numeric_limits<uint64_t>::max() - library_headroom_bytes ||
        o.min_free_bytes > std::numeric_limits<uint64_t>::max() - library_headroom_bytes - o.max_bytes) {
        throw std::runtime_error("memory budget overflows");
    }
    return o;
}

uint64_t case_bytes(int n, const std::string & dtype) {
    const uint64_t elements = uint64_t(n) * n;
    return elements * (dtype == "int8" ? 6 : 8) + workspace_bytes;
}

uint64_t peak_bytes(const Options & o) {
    uint64_t peak = 0;
    for (int n : o.sizes) for (const auto & type : o.dtypes) peak = std::max(peak, case_bytes(n, type));
    if (peak > o.max_bytes) throw std::runtime_error("planned explicit device allocation " + std::to_string(peak) + " exceeds --max-bytes");
    return peak;
}

void cuda_check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status, const char * operation) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(int(status)));
}

struct Memory {
    uint64_t current = 0;
    uint64_t peak = 0;
    uint64_t baseline_free = 0;
    uint64_t max_observed_growth = 0;
    uint64_t min_observed_free = std::numeric_limits<uint64_t>::max();
    uint64_t observe() {
        size_t free = 0, total = 0;
        cuda_check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
        min_observed_free = std::min(min_observed_free, uint64_t(free));
        if (baseline_free > free) max_observed_growth = std::max(max_observed_growth, baseline_free - free);
        return free;
    }
};

struct Buffer {
    void * ptr = nullptr;
    uint64_t bytes;
    Memory & memory;
    Buffer(uint64_t n, Memory & m) : bytes(n), memory(m) {
        cuda_check(cudaMalloc(&ptr, n), "cudaMalloc (no retry)");
        memory.current += bytes;
        memory.peak = std::max(memory.peak, memory.current);
        memory.observe();
    }
    ~Buffer() { if (ptr) { cudaFree(ptr); memory.current -= bytes; } }
    Buffer(const Buffer &) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { cuda_check(cudaEventCreate(&value), "cudaEventCreate"); }
    ~Event() { if (value) cudaEventDestroy(value); }
};
struct Blas {
    cublasHandle_t value = nullptr;
    Blas() { blas_check(cublasCreate(&value), "cublasCreate"); }
    ~Blas() { if (value) cublasDestroy(value); }
};

// Small, exactly representable values keep every FP32 partial sum integral and
// below 2^24. They also avoid INT32 overflow: |sum| <= 16*K <= 131072.
int input_value(uint64_t index, uint64_t seed) {
    uint64_t x = index + seed;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return int((x ^ (x >> 31)) % 9) - 4;
}
constexpr uint64_t seed_a = 0x243f6a8885a308d3ULL;
constexpr uint64_t seed_b = 0x13198a2e03707344ULL;

uint16_t encoded_value(int value, bool bf16) {
    constexpr uint16_t half_positive[]{0x0000, 0x3c00, 0x4000, 0x4200, 0x4400};
    constexpr uint16_t bf_positive[]{0x0000, 0x3f80, 0x4000, 0x4040, 0x4080};
    const auto magnitude = value < 0 ? -value : value;
    return (bf16 ? bf_positive[magnitude] : half_positive[magnitude]) | (value < 0 ? 0x8000 : 0);
}

void initialize(Buffer & buffer, uint64_t count, const std::string & dtype, uint64_t seed) {
    constexpr uint64_t chunk_elements = 1024 * 1024;
    if (dtype == "int8") {
        std::vector<int8_t> chunk(std::min(count, chunk_elements));
        for (uint64_t off = 0; off < count; off += chunk_elements) {
            const auto length = std::min(chunk_elements, count - off);
            for (uint64_t j = 0; j < length; ++j) chunk[j] = int8_t(input_value(off + j, seed));
            cuda_check(cudaMemcpy(static_cast<int8_t *>(buffer.ptr) + off, chunk.data(), length, cudaMemcpyHostToDevice), "initialize INT8 input");
        }
    } else {
        std::vector<uint16_t> chunk(std::min(count, chunk_elements));
        for (uint64_t off = 0; off < count; off += chunk_elements) {
            const auto length = std::min(chunk_elements, count - off);
            for (uint64_t j = 0; j < length; ++j) chunk[j] = encoded_value(input_value(off + j, seed), dtype == "bf16");
            cuda_check(cudaMemcpy(static_cast<uint16_t *>(buffer.ptr) + off, chunk.data(), length * sizeof(uint16_t), cudaMemcpyHostToDevice), "initialize floating input");
        }
    }
}

std::string driver_string() {
    // Optional NVML query supplies the NVIDIA driver release string. The CUDA
    // driver API version below is a different value. No clock/power writes.
    void * library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) return {};
    using Init = int (*)();
    using Version = int (*)(char *, unsigned int);
    auto init = reinterpret_cast<Init>(dlsym(library, "nvmlInit_v2"));
    auto version = reinterpret_cast<Version>(dlsym(library, "nvmlSystemGetDriverVersion"));
    auto shutdown = reinterpret_cast<Init>(dlsym(library, "nvmlShutdown"));
    char text[128]{};
    std::string result;
    if (init && version && shutdown && init() == 0) {
        if (version(text, sizeof(text)) == 0) result = text;
        shutdown();
    }
    dlclose(library);
    return result;
}

std::string uuid_string(const cudaUUID_t & uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
        out << std::setw(2) << unsigned(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

std::string run_case(const Options & o, int n, const std::string & dtype,
                     Blas & blas, Stream & stream, Buffer & workspace, Memory & memory) {
    const auto required = case_bytes(n, dtype) - workspace_bytes;
    if (memory.observe() < required + o.min_free_bytes) throw std::runtime_error("free VRAM preflight failed for " + dtype + "/" + std::to_string(n));
    const uint64_t elements = uint64_t(n) * n;
    const uint64_t input_bytes = elements * (dtype == "int8" ? 1 : 2);
    Buffer a(input_bytes, memory), b(input_bytes, memory), c(elements * 4, memory);
    initialize(a, elements, dtype, seed_a);
    initialize(b, elements, dtype, seed_b);
    cuda_check(cudaMemsetAsync(c.ptr, 0, elements * 4, stream.value), "initialize output");
    blas_check(cublasSetWorkspace(blas.value, workspace.ptr, workspace_bytes), "cublasSetWorkspace");
    const auto math = static_cast<cublasMath_t>(unsigned(CUBLAS_DEFAULT_MATH) | unsigned(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    blas_check(cublasSetMathMode(blas.value, math), "cublasSetMathMode");
    const float alpha_f = 1.0f, beta_f = 0.0f;
    const int32_t alpha_i = 1, beta_i = 0;
    const bool integer = dtype == "int8";
    const auto input_type = integer ? CUDA_R_8I : (dtype == "bf16" ? CUDA_R_16BF : CUDA_R_16F);
    const auto output_type = integer ? CUDA_R_32I : CUDA_R_32F;
    const auto compute = integer ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F;
    auto gemm = [&] {
        // Column-major TN: stored A[K,M], B[K,N], C[M,N]. Dimensions and
        // leading dimensions are multiples of 4; cudaMalloc alignment >=256B.
        blas_check(cublasGemmEx(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
            integer ? static_cast<const void *>(&alpha_i) : static_cast<const void *>(&alpha_f),
            a.ptr, input_type, n, b.ptr, input_type, n,
            integer ? static_cast<const void *>(&beta_i) : static_cast<const void *>(&beta_f),
            c.ptr, output_type, n, compute, CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
    };
    for (int i = 0; i < o.warmups; ++i) gemm();
    cuda_check(cudaStreamSynchronize(stream.value), "warmup synchronize");
    if (memory.observe() < o.min_free_bytes) throw std::runtime_error("free VRAM reserve exhausted during warmup");
    Event start, stop;
    std::vector<double> samples;
    for (int sample = 0; sample < o.samples; ++sample) {
        cuda_check(cudaEventRecord(start.value, stream.value), "start event");
        for (int iteration = 0; iteration < o.iterations; ++iteration) gemm();
        cuda_check(cudaEventRecord(stop.value, stream.value), "stop event");
        cuda_check(cudaEventSynchronize(stop.value), "timing synchronize");
        float elapsed = 0;
        cuda_check(cudaEventElapsedTime(&elapsed, start.value, stop.value), "elapsed time");
        if (!std::isfinite(elapsed) || elapsed <= 0) throw std::runtime_error("invalid event duration");
        samples.push_back(double(elapsed) / o.iterations);
        if (memory.observe() < o.min_free_bytes) throw std::runtime_error("free VRAM reserve exhausted during measurement");
    }
    // Sparse, deterministic exact references include edges and interior output
    // locations. No full output allocation/download or validation in timing.
    constexpr int checked = 32;
    int64_t reference_checksum = 0;
    for (int i = 0; i < checked; ++i) {
        const int row = i == 0 ? 0 : (i == 1 ? n - 1 : int((uint64_t(i) * 2654435761ULL) % n));
        const int col = i == 0 ? 0 : (i == 1 ? n - 1 : int((uint64_t(i) * 2246822519ULL + 3266489917ULL) % n));
        int64_t expected = 0;
        for (int k = 0; k < n; ++k) expected += int64_t(input_value(uint64_t(k) + uint64_t(row) * n, seed_a)) * input_value(uint64_t(k) + uint64_t(col) * n, seed_b);
        const uint64_t offset = uint64_t(row) + uint64_t(col) * n;
        bool matches = false;
        if (integer) {
            int32_t actual = 0;
            cuda_check(cudaMemcpy(&actual, static_cast<int32_t *>(c.ptr) + offset, sizeof(actual), cudaMemcpyDeviceToHost), "validate integer output");
            matches = actual == expected;
        } else {
            float actual = 0;
            cuda_check(cudaMemcpy(&actual, static_cast<float *>(c.ptr) + offset, sizeof(actual), cudaMemcpyDeviceToHost), "validate floating output");
            matches = std::isfinite(actual) && double(actual) == double(expected);
        }
        if (!matches) throw std::runtime_error("exact sampled output validation failed for " + dtype + "/" + std::to_string(n));
        reference_checksum += expected;
    }
    auto ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    const auto median = ordered.size() % 2 ? ordered[ordered.size() / 2] : (ordered[ordered.size() / 2 - 1] + ordered[ordered.size() / 2]) / 2;
    const auto tops = (2.0 * n * n * n) / (median * 1.0e9);
    std::ostringstream out;
    out << std::setprecision(12) << "{\"input_dtype\":" << quoted(dtype)
        << ",\"output_dtype\":" << quoted(integer ? "int32" : "fp32")
        << ",\"accumulation_dtype\":" << quoted(integer ? "int32" : "fp32")
        << ",\"compute_type\":" << quoted(integer ? "CUBLAS_COMPUTE_32I" : "CUBLAS_COMPUTE_32F")
        << ",\"math_mode\":\"CUBLAS_DEFAULT_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION\""
        << ",\"algorithm\":\"CUBLAS_GEMM_DEFAULT\",\"tensor_core_kernel_verified\":false"
        << ",\"layout\":\"column_major_TN\",\"m\":" << n << ",\"n\":" << n << ",\"k\":" << n
        << ",\"alpha\":1,\"beta\":0,\"warmups\":" << o.warmups
        << ",\"iterations_per_sample\":" << o.iterations << ",\"samples_ms\":[";
    for (size_t i = 0; i < samples.size(); ++i) out << (i ? "," : "") << samples[i];
    out << "],\"median_ms\":" << median << ",\"tera_operations_per_second\":" << tops
        << ",\"operations_unit\":" << quoted(integer ? "TOPS" : "TFLOPS")
        << ",\"explicit_device_bytes\":" << case_bytes(n, dtype)
        << ",\"validation\":{\"status\":\"passed\",\"method\":\"32_exact_sampled_integer_dot_products\",\"checked_outputs\":" << checked
        << ",\"reference_checksum\":" << reference_checksum << "}}";
    return out.str();
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const auto o = parse(argc, argv);
        const auto planned_peak = peak_bytes(o);
        if (o.plan) {
            std::cout << "{\"schema\":" << quoted(schema) << ",\"status\":\"plan\",\"cuda_called\":false,\"planned_peak_explicit_device_bytes\":" << planned_peak
                << ",\"max_bytes\":" << o.max_bytes << ",\"workspace_bytes\":" << workspace_bytes
                << ",\"min_free_bytes\":" << o.min_free_bytes << ",\"library_preflight_headroom_bytes\":" << library_headroom_bytes << "}\n";
            return 0;
        }
        cuda_check(cudaSetDevice(o.device), "cudaSetDevice");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, o.device), "cudaGetDeviceProperties");
        char pci_bus_id[32]{};
        cuda_check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), o.device), "cudaDeviceGetPCIBusId");
        if (properties.major < 8) throw std::runtime_error("probe requires SM80+ for BF16 and matching integer diagnostics");
        Memory memory;
        memory.baseline_free = memory.observe();
        if (memory.baseline_free < planned_peak + o.min_free_bytes + library_headroom_bytes) throw std::runtime_error("initial free VRAM preflight failed");
        Stream stream;
        Blas blas;
        blas_check(cublasSetStream(blas.value, stream.value), "cublasSetStream");
        blas_check(cublasSetPointerMode(blas.value, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        int library_version = 0, driver_api_version = 0, runtime_version = 0;
        blas_check(cublasGetVersion(blas.value, &library_version), "cublasGetVersion");
        cuda_check(cudaDriverGetVersion(&driver_api_version), "cudaDriverGetVersion");
        cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
        Buffer workspace(workspace_bytes, memory);
        std::vector<std::string> results;
        for (int n : o.sizes) for (const auto & dtype : o.dtypes) results.push_back(run_case(o, n, dtype, blas, stream, workspace, memory));
        std::ostringstream out;
        const auto driver = driver_string();
        out << "{\"schema\":" << quoted(schema) << ",\"status\":\"passed\",\"device\":" << o.device
            << ",\"gpu_name\":" << quoted(properties.name) << ",\"gpu_uuid\":" << quoted(uuid_string(properties.uuid))
            << ",\"pci_bus_id\":" << quoted(pci_bus_id)
            << ",\"compute_capability\":" << quoted(std::to_string(properties.major) + "." + std::to_string(properties.minor))
            << ",\"nvidia_driver\":" << (driver.empty() ? "null" : quoted(driver))
            << ",\"cuda_driver_api_version\":" << driver_api_version << ",\"cuda_runtime_version\":" << runtime_version
            << ",\"cuda_compile_version\":" << CUDART_VERSION << ",\"cublas_version\":" << library_version
            << ",\"device_total_memory_bytes\":" << properties.totalGlobalMem << ",\"free_memory_before_library_bytes\":" << memory.baseline_free
            << ",\"planned_peak_explicit_device_bytes\":" << planned_peak << ",\"max_allocated_bytes\":" << memory.peak
            << ",\"allocation_cap_bytes\":" << o.max_bytes << ",\"workspace_bytes\":" << workspace_bytes
            << ",\"min_free_reserve_bytes\":" << o.min_free_bytes << ",\"min_observed_free_bytes\":" << memory.min_observed_free
            << ",\"max_observed_device_memory_growth_bytes\":" << memory.max_observed_growth
            << ",\"memory_growth_includes_other_processes\":true,\"clocks_power_capture\":\"external_required\",\"results\":[";
        for (size_t i = 0; i < results.size(); ++i) out << (i ? "," : "") << results[i];
        out << "]}\n";
        std::cout << out.str();
        return 0;
    } catch (const std::exception & error) {
        std::cout << "{\"schema\":" << quoted(schema) << ",\"status\":\"error\",\"error\":" << quoted(error.what()) << "}\n";
        return 1;
    }
}
