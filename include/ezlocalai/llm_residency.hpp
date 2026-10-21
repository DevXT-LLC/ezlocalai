#pragma once
#include "ezlocalai/llm_catalog.hpp"
#include "ezlocalai/resource_manager.hpp"
#include <functional>

namespace ezlocalai {
// Each preload failure is isolated. Only generated replicas become unavailable;
// configured originals remain eligible for a later request to retry loading.
std::set<std::string> preload_llm_instances(const LlmCatalog &catalog, bool swap,
    const std::function<void(const std::string &)> &load,
    const std::function<void(const std::string &, const std::string &)> &failed);
// The estimator is lazy: forced modes, a single model, and CPU-only workers
// must not acquire model files merely to choose their residency mode.
bool select_llm_swap_residency(const LlmCatalog &catalog,
    const std::vector<double> &gpu_capacity_gb, const std::string &requested,
    const std::string &margin_gb, const std::function<double(const std::string &)> &estimate);
struct LlmHeadroomOperations {
    std::function<double()> estimate;
    std::function<std::vector<double>()> free_per_gpu;
    std::function<double()> total_free;
    std::function<std::string()> margin;
};
struct LlmHeadroom {
    double estimate_gb = 8, free_gb = 0, required_gb = 0, reclaim_gb = 0;
};
LlmHeadroom llm_load_headroom(int64_t main_gpu, const LlmHeadroomOperations &operations);
// Selection includes the reference's ignored types (vision LLM and music).
// Those entries count toward the LRU estimate but have no cleanup dispatch.
void reclaim_llm_auxiliary_memory(ResourceManager &resources, double required_gb,
    const std::function<void(ModelType)> &destroy);
}
