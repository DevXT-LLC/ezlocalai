#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ezlocalai {
struct LlmLoadConfiguration {
    std::string model;
    int64_t context = 16384, initial_ubatch = 256, total_layers = 41;
    std::optional<int64_t> gpu_layers;
};
struct LlmLoadSelection { std::optional<int64_t> gpu_layers, ubatch; };
struct LlmLoadOperations {
    std::function<void(std::optional<int64_t>, std::optional<int64_t>)> load;
    std::function<bool(int64_t, int64_t)> probe;
    std::function<void()> cleanup;
    // The reference probes current free memory only after the first failure.
    std::function<int64_t()> configured_ubatch;
};
bool retryable_llm_load_error(const std::string &message);
std::vector<int64_t> reduced_llm_ubatches(int64_t initial);
LlmLoadSelection load_llm_resilient(const LlmLoadConfiguration &configuration, const LlmLoadOperations &operations);
}
