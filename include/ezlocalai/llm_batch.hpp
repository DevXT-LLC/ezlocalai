#pragma once
#include "ezlocalai/video_runtime.hpp"
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace ezlocalai {
struct LlmBatchProbes {
    std::function<bool()> available;
    std::function<int()> count;
    std::function<CudaMemoryInfo(int)> memory;
    std::function<std::pair<int, int>(int)> capability;
    std::function<CudaDevice(int64_t)> properties;
};
struct GpuInferenceProfile { std::string family; double capacity_gb = 0; };
struct LlmAutoBatch { int64_t batch = 1024, ubatch = 256; std::string reason; };
// Canonical decimal strings retain Python integer precision until the native
// engine boundary. Configuration selection itself only compares integers.
struct LlmBatchParameters { std::string batch, ubatch; };
LlmBatchProbes native_llm_batch_probes();
GpuInferenceProfile gpu_inference_profile(const LlmBatchProbes &probes, std::optional<int64_t> main_gpu = 0);
LlmAutoBatch calculate_llm_auto_batch(int64_t main_gpu, int64_t context, const std::string &model,
    const LlmBatchProbes &probes = native_llm_batch_probes());
LlmBatchParameters resolve_llm_batch_parameters(const LlmAutoBatch &automatic,
    const std::string &batch_env, const std::string &ubatch_env,
    std::optional<int64_t> batch_override = {}, std::optional<int64_t> ubatch_override = {});
int64_t configured_llm_ubatch(const std::string &raw, int64_t automatic, int64_t context);
}
