#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ezlocalai {
struct LlmParameterError : std::runtime_error {
    std::string category, message;
    LlmParameterError(std::string type, const std::string &text) : std::runtime_error(text), category(std::move(type)), message(text) {}
};
std::string llm_parameter_integer(const std::string &raw);
void validate_llm_reasoning_budget(const std::string &raw);
int32_t llm_parallel_parameter(int64_t requested, int64_t context, bool speculative);
int llm_image_min_tokens(const std::string &model);
struct LlmTensorParameters { bool split_none = false, fit = true; };
// CommonParams.tensor_split returns a list copy in the pinned binding. The
// reference's per-element writes do not change the native array.
LlmTensorParameters llm_tensor_parameters(const std::vector<double> &split, int gpu_count, bool initial_fit = true);
}
