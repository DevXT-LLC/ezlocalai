#include "ezlocalai/llm_load.hpp"
#include "ezlocalai/llm_slots.hpp"
#include <algorithm>
#include <stdexcept>

namespace ezlocalai {
bool retryable_llm_load_error(const std::string &message) {
    const auto lower = llm_unicode_lower(message);
    for (const auto marker : {"out of memory", "cuda", "vram", "gpu", "allocat", "memory", "resource",
                             "failed to init server", "failed to initialize llm server", "failed to create context", "failed to create_context"})
        if (lower.find(marker) != std::string::npos) return true;
    return false;
}
std::vector<int64_t> reduced_llm_ubatches(int64_t initial) {
    std::vector<int64_t> values;
    for (auto current = std::max<int64_t>(1, initial); current > 128;) { current = std::max<int64_t>(128, current / 2); values.push_back(current); }
    return values;
}
LlmLoadSelection load_llm_resilient(const LlmLoadConfiguration &config, const LlmLoadOperations &operations) {
    std::string last_error;
    try { operations.load(config.gpu_layers, std::nullopt); return {config.gpu_layers, std::nullopt}; }
    catch (const std::exception &error) {
        if (!retryable_llm_load_error(error.what()) || config.gpu_layers == 0) throw;
        last_error = error.what();
    }
    const auto initial_ubatch = operations.configured_ubatch ? operations.configured_ubatch() : config.initial_ubatch;
    const auto candidates = reduced_llm_ubatches(initial_ubatch);
    for (const auto ubatch : candidates) {
        operations.cleanup();
        try { operations.load(config.gpu_layers, ubatch); return {config.gpu_layers, ubatch}; }
        catch (const std::exception &error) {
            if (!retryable_llm_load_error(error.what())) throw;
            last_error = error.what();
        }
    }
    const auto requested = !config.gpu_layers || *config.gpu_layers < 0 ? config.total_layers : std::min(config.total_layers, *config.gpu_layers);
    int64_t low = 0, high = requested > 0 ? requested - 1 : 0;
    std::optional<int64_t> best;
    const auto safe_ubatch = candidates.empty() ? initial_ubatch : candidates.back();
    while (low <= high) {
        const auto layers = low + (high - low) / 2;
        operations.cleanup();
        if (operations.probe(layers, safe_ubatch)) { best = layers; low = layers + 1; }
        else high = layers - 1;
    }
    if (!best) throw std::runtime_error("Failed to allocate inference capacity for " + config.model + " at n_ctx=" + std::to_string(config.context) + ": " + last_error);
    operations.cleanup();
    operations.load(best, safe_ubatch);
    return {best, safe_ubatch};
}
}
