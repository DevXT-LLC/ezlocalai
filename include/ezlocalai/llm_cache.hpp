#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace ezlocalai {
struct LlmCacheError : std::runtime_error {
    std::string category;
    LlmCacheError(std::string type, const std::string &message) : std::runtime_error(message), category(std::move(type)) {}
};
struct PromptCache { std::string mib, reason; };
struct PromptCacheRam { int64_t available_mib, total_mib; int64_t swap_available_mib = 0; };
struct PromptCacheBudget { PromptCache requested, effective; int64_t swap_allowance_mib = 0; };
using LlmCacheEnvironment = std::function<std::string(const std::string &, const std::string &)>;
std::string resolve_llm_kv_cache_type(const std::string &global, const std::string &card);
std::string configured_llm_kv_cache_type(const std::string &family, const LlmCacheEnvironment &environment, const std::string &model = "");
PromptCache resolve_llm_prompt_cache(const std::optional<std::string> &raw, int64_t context);
PromptCache cap_llm_prompt_cache(PromptCache requested, const LlmCacheEnvironment &environment,
    const std::function<PromptCacheRam()> &memory, std::optional<int64_t> available = {}, std::optional<int64_t> total = {});
// Linux psutil virtual_memory's total/available subset, including older-kernel
// and zero-MemAvailable fallbacks. The zone reader is called only when needed.
PromptCacheRam prompt_cache_ram_from_proc(const std::string &meminfo,
    const std::function<std::optional<std::string>()> &zoneinfo, int64_t page_size);
PromptCacheRam native_prompt_cache_ram();
// Apply every visible cgroup-v2 ancestor limit, including limits inherited
// through a parent slice. Paths are relative to the mounted controller root.
PromptCacheRam prompt_cache_ram_with_cgroup(PromptCacheRam ram,
    const std::string &root, const std::string &relative_path);
// Native policy extends the frozen Python helpers above: larger automatic
// ceilings on high-RAM hosts and optional, bounded existing swap capacity.
PromptCacheBudget native_llm_prompt_cache_budget(const std::optional<std::string> &raw, int64_t context,
    const LlmCacheEnvironment &environment, const std::function<PromptCacheRam()> &memory);
LlmCacheEnvironment native_llm_cache_environment();
}
