#pragma once
#include "ezlocalai/llm_batch.hpp"
#include "ezlocalai/hub_assets.hpp"
#include "ezlocalai/llm_parameters.hpp"
#include <stdexcept>

namespace ezlocalai {
inline constexpr const char *dflash_repository = "incoai/Qwen3.8-27B-DFlash2-GGUF";
inline constexpr const char *dflash_filename = "Qwen3.8-27B-DFlash2-Q4_K_M.gguf";
inline constexpr const char *dflash_revision = "51962825493a48b846b40126d35c799ac4093ad0";
using LlmSpeculativeEnvironment = std::function<std::string(const std::string &, const std::string &)>;
using LlmSpeculativeError = LlmParameterError;
struct LlmDraftSettings { int n_max = 0; double p_min = 0, card_vram_gb = 0; };
double llm_model_size_billions(const std::string &model);
std::pair<int, double> llm_mtp_draft_length(int64_t main_gpu, const std::string &model,
    const LlmSpeculativeEnvironment &environment, const LlmBatchProbes &probes = native_llm_batch_probes());
double llm_mtp_draft_probability(const std::string &model, const LlmSpeculativeEnvironment &environment);
LlmDraftSettings llm_dflash_settings(int64_t main_gpu, const LlmSpeculativeEnvironment &environment,
    const LlmBatchProbes &probes = native_llm_batch_probes());
struct DflashDownloadHooks {
    std::function<bool(const std::string &)> is_file;
    std::function<std::string(const HubAsset &)> download;
};
DflashDownloadHooks native_dflash_download_hooks();
std::string download_dflash_model(const LlmSpeculativeEnvironment &environment,
    const DflashDownloadHooks &hooks = native_dflash_download_hooks());
}
