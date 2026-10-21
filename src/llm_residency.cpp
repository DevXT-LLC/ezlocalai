#include "ezlocalai/llm_residency.hpp"
#include "ezlocalai/configuration.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ezlocalai {
std::set<std::string> preload_llm_instances(const LlmCatalog &catalog, bool swap,
    const std::function<void(const std::string &)> &load,
    const std::function<void(const std::string &, const std::string &)> &failed) {
    std::set<std::string> unavailable;
    for (const auto &id : catalog.available) {
        try { load(id); }
        catch (const std::exception &error) {
            if (catalog.automatic.count(id)) unavailable.insert(id);
            failed(id, error.what());
        }
        if (swap) break;
    }
    return unavailable;
}
LlmHeadroom llm_load_headroom(int64_t main_gpu, const LlmHeadroomOperations &operations) {
    LlmHeadroom result;
    try { result.estimate_gb = operations.estimate(); } catch (const std::exception &) {}
    const auto free = operations.free_per_gpu();
    result.free_gb = main_gpu >= 0 && main_gpu < static_cast<int64_t>(free.size())
        ? free[static_cast<size_t>(main_gpu)] : operations.total_free();
    double margin = 1.5;
    if (!parse_configuration_float(operations.margin(), &margin)) margin = 1.5;
    result.required_gb = result.estimate_gb + std::max(0.0, margin);
    if (result.free_gb < result.required_gb) result.reclaim_gb = result.required_gb - result.free_gb;
    return result;
}
void reclaim_llm_auxiliary_memory(ResourceManager &resources, double required_gb,
    const std::function<void(ModelType)> &destroy) {
    const auto selected = resources.get_models_to_unload(required_gb);
    if (selected.empty()) return;
    for (const auto type : selected) switch (type) {
        case ModelType::Tts: case ModelType::Stt: case ModelType::Image:
        case ModelType::Video: case ModelType::Embedding: destroy(type); break;
        default: break;
    }
    resources.release_unused_cuda_memory();
    // Preserve the reference's final live probe and exception boundary.
    resources.get_total_free_vram();
}
bool select_llm_swap_residency(const LlmCatalog &catalog,
    const std::vector<double> &gpu_capacity_gb, const std::string &requested,
    const std::string &margin_raw, const std::function<double(const std::string &)> &estimate) {
    auto mode = strip_config_value(requested);
    for (auto &c : mode) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    if (catalog.available.size() <= 1) return false;
    if (mode == "swap" || mode == "resident") return mode == "swap";
    if (gpu_capacity_gb.empty()) return false;
    std::map<int64_t, double> by_gpu;
    bool failed = false;
    for (const auto &model : catalog.available) {
        try {
            const double value = estimate(model);
            const auto config = catalog.configs.find(model);
            by_gpu[config == catalog.configs.end() ? 0 : config->second.main_gpu] += value;
        } catch (const std::exception &) { failed = true; }
    }
    double margin = 1.5;
    if (!parse_configuration_float(margin_raw, &margin)) margin = 1.5;
    margin = std::max(0.0, margin);
    bool overcommitted = false;
    for (const auto &[gpu, value] : by_gpu) {
        double capacity;
        if (gpu < static_cast<int64_t>(gpu_capacity_gb.size())) {
            const auto index = gpu < 0 ? gpu + static_cast<int64_t>(gpu_capacity_gb.size()) : gpu;
            capacity = gpu_capacity_gb.at(static_cast<size_t>(index));
        } else capacity = std::accumulate(gpu_capacity_gb.begin(), gpu_capacity_gb.end(), 0.0);
        if (value > std::max(0.0, capacity - margin)) overcommitted = true;
    }
    return overcommitted && !failed;
}
}
