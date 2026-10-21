#pragma once
#include "ezlocalai/llm_slots.hpp"
#include <array>
#include <set>
#include <stdexcept>

namespace ezlocalai {
struct LlmCatalogSettings {
    std::string models, main_gpus = "0", contexts = "65536", parallels = "1", quants = "Q4_K_XL", speculative_type = "auto";
    int gpu_count = 0;
};
struct LlmModelConfiguration {
    int64_t main_gpu = 0, context = 65536, parallel = 1;
    std::optional<std::string> quant;
    std::optional<std::array<double,128>> tensor_split;
    size_t original_index = 0;
    std::string speculative_type;
};
struct LlmCatalogError : std::runtime_error {
    std::string type;
    LlmCatalogError(std::string kind, const std::string &message) : std::runtime_error(message), type(std::move(kind)) {}
};
struct LlmCatalog {
    std::vector<std::string> available;
    std::map<std::string,std::string> sources;
    std::vector<std::pair<std::string,std::vector<std::string>>> groups;
    std::map<std::string,LlmModelConfiguration> configs;
    std::set<std::string> automatic;
    std::string source(const std::string &id) const;
    std::optional<std::string> resolve(const std::string &request) const;
    std::optional<std::string> slot_model(const std::string &request) const;
    std::vector<LlmReplica> replicas() const;
    std::vector<std::string> public_models() const;
};
std::string llm_speculative_backend(const std::string &model, const std::string &setting);
LlmCatalog parse_llm_catalog(const LlmCatalogSettings &settings);
}
