#include "ezlocalai/llm_cache.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <limits>
#include <map>
#include <sstream>
#include <unistd.h>

namespace ezlocalai {
namespace {
std::string setting(const std::string &raw) { return llm_unicode_lower(strip_config_value(raw)); }
std::string minimum(const std::string &a, const std::string &b) { return configuration_integer_less(a, b) ? a : b; }
std::string grouped(int64_t number) {
    auto value = std::to_string(number);
    for (int i = static_cast<int>(value.size()) - 3; i > (number < 0 ? 1 : 0); i -= 3) value.insert(i, ",");
    return value;
}
std::string fixed(double number, int precision) {
    char buffer[512]; const auto result = std::to_chars(buffer, buffer + sizeof(buffer), number, std::chars_format::fixed, precision);
    if (result.ec != std::errc{}) throw std::runtime_error("Prompt-cache number formatting failed");
    return {buffer, result.ptr};
}
}
std::string resolve_llm_kv_cache_type(const std::string &global, const std::string &card) {
    const auto value = setting(card.empty() ? global : card);
    if (value.empty() || value == "auto") return "q4_0";
    if (value == "q4_0" || value == "q8_0" || value == "f16" || value == "f32") return value;
    throw LlmCacheError("ValueError", "KV_CACHE_TYPE must be auto, q4_0, q8_0, f16 or f32");
}
std::string configured_llm_kv_cache_type(const std::string &family, const LlmCacheEnvironment &environment) {
    const auto card = family.empty() ? "" : environment("KV_CACHE_TYPE_" + family, "");
    return resolve_llm_kv_cache_type(card.empty() ? environment("KV_CACHE_TYPE", "auto") : "", card);
}
PromptCache resolve_llm_prompt_cache(const std::optional<std::string> &raw, int64_t context) {
    const auto value = setting(raw.value_or("auto"));
    if (value.empty() || value == "auto") {
        const auto tokens = std::max<int64_t>(0, context);
        const int64_t scaled = tokens >= 500000 ? 32768 : ((tokens / 16 + 1023) / 1024) * 1024;
        return {std::to_string(std::min<int64_t>(32768, std::max<int64_t>(8192, scaled))), "auto for " + grouped(context) + " context tokens"};
    }
    for (const auto disabled : {"0", "false", "no", "off", "disabled", "none"})
        if (value == disabled) return {"0", "disabled by LLM_PROMPT_CACHE_MIB"};
    const auto parsed = parse_configuration_integer(value);
    if (!parsed) return {"8192", "invalid LLM_PROMPT_CACHE_MIB=" + (raw ? python_string_repr(*raw) : "None") + "; using safe auto minimum"};
    if (!configuration_integer_less("0", *parsed)) return {"0", "disabled by non-positive LLM_PROMPT_CACHE_MIB"};
    return {*parsed, "explicit LLM_PROMPT_CACHE_MIB"};
}
PromptCache cap_llm_prompt_cache(PromptCache requested, const LlmCacheEnvironment &environment,
    const std::function<PromptCacheRam()> &memory, std::optional<int64_t> available, std::optional<int64_t> total) {
    if (!configuration_integer_less("0", requested.mib)) return requested;
    const auto unsafe = setting(environment("LLM_PROMPT_CACHE_ALLOW_UNSAFE", "false"));
    if (unsafe == "1" || unsafe == "true" || unsafe == "yes" || unsafe == "on") {
        requested.reason += "; unsafe RAM cap bypass enabled"; return requested;
    }
    if (!available || !total) {
        try { const auto ram = memory(); available = ram.available_mib; total = ram.total_mib; }
        catch (const LlmCacheError &error) {
            if (error.category != "TypeError" && error.category != "ValueError" && error.category != "ImportError"
                && error.category != "ModuleNotFoundError" && error.category != "UnicodeError"
                && error.category != "UnicodeDecodeError" && error.category != "UnicodeEncodeError" && error.category != "UnicodeTranslateError") throw;
            return {minimum(requested.mib, "4096"), requested.reason + "; RAM safety probe unavailable, capped at 4096 MiB"};
        }
    }
    auto margin = parse_configuration_integer(environment("LLM_PROMPT_CACHE_RAM_MARGIN_MIB", "4096")).value_or("4096");
    if (configuration_integer_less(margin, "0")) margin = "0";
    double fraction;
    if (!parse_configuration_float(environment("LLM_PROMPT_CACHE_MAX_RAM_FRACTION", "0.25"), &fraction)) fraction = .25;
    // Preserve Python's ordered min/max behavior, including NaN and -0.0.
    if (fraction < 0) fraction = 0;
    if (fraction > 1) fraction = 1;
    if (std::isnan(fraction)) throw LlmCacheError("ValueError", "cannot convert float NaN to integer");
    int64_t budget = 0;
    if (*available > 0 && configuration_integer_less(margin, std::to_string(*available))) budget = *available - std::stoll(margin);
    const auto fraction_budget = fixed(std::trunc(static_cast<double>(*total) * fraction), 0);
    const auto normalized_fraction = parse_configuration_integer(fraction_budget).value_or("0");
    auto safe = minimum(std::to_string(budget), normalized_fraction);
    if (configuration_integer_less(safe, "0")) safe = "0";
    safe = std::to_string((std::stoll(safe) / 256) * 256);
    if (!configuration_integer_less(safe, requested.mib)) return requested;
    return {safe, requested.reason + "; RAM safety cap (available=" + std::to_string(*available) + " MiB, total="
        + std::to_string(*total) + " MiB, reserve=" + margin + " MiB, max_fraction=" + fixed(fraction, 2) + ')'};
}
PromptCacheRam prompt_cache_ram_from_proc(const std::string &meminfo,
    const std::function<std::optional<std::string>()> &zoneinfo, int64_t page_size) {
    std::map<std::string, int64_t> fields;
    std::istringstream input(meminfo);
    for (std::string line; std::getline(input, line);) {
        std::istringstream row(line); std::string key; int64_t value;
        if (!(row >> key >> value)) throw LlmCacheError("ValueError", "Invalid memory statistics");
        fields[key] = value * 1024;
    }
    const auto required = [&](const std::string &key) {
        const auto found = fields.find(key);
        if (found == fields.end()) throw LlmCacheError("KeyError", "b'" + key + "'");
        return found->second;
    };
    const auto total = required("MemTotal:"), free = required("MemFree:");
    auto available = fields.count("MemAvailable:") ? fields.at("MemAvailable:") : 0;
    if (!available) {
        available = free + (fields.count("Cached:") ? fields.at("Cached:") : 0);
        if (fields.count("Active(file):") && fields.count("Inactive(file):") && fields.count("SReclaimable:")) {
            if (const auto zones = zoneinfo()) {
                int64_t low = 0; std::istringstream lines(*zones);
                for (std::string line; std::getline(lines, line);) {
                    line = strip_config_value(line);
                    if (line.compare(0, 3, "low") == 0) { std::istringstream row(line); std::string key; int64_t value; if (!(row >> key >> value)) throw LlmCacheError("ValueError", "Invalid zone memory statistics"); low += value; }
                }
                const auto watermark = low * page_size;
                const auto pages = fields.at("Active(file):") + fields.at("Inactive(file):"), slab = fields.at("SReclaimable:");
                double value = static_cast<double>(free - watermark);
                value += pages - std::min(pages / 2.0, static_cast<double>(watermark));
                value += slab - std::min(slab / 2.0, static_cast<double>(watermark));
                available = static_cast<int64_t>(value);
            }
        }
    }
    if (available < 0) available = 0;
    else if (available > total) available = free;
    return {static_cast<int64_t>(available / 1048576.0), static_cast<int64_t>(total / 1048576.0),
        fields.count("SwapFree:") ? std::max<int64_t>(0, fields.at("SwapFree:") / 1048576) : 0};
}
PromptCacheRam prompt_cache_ram_with_cgroup(PromptCacheRam ram,
    const std::string &root, const std::string &relative_path) {
    namespace fs = std::filesystem;
    const auto base = fs::path(root).lexically_normal();
    auto relative = fs::path(relative_path).relative_path().lexically_normal();
    // A cgroup namespace may hide the process's host path. Never escape the
    // mounted controller; its root still exposes the namespace's limits.
    if (std::find(relative.begin(), relative.end(), fs::path("..")) != relative.end()) relative.clear();
    auto current = base / relative;
    if (!fs::exists(current)) current = base;
    const auto number = [](const fs::path &path) -> std::optional<int64_t> {
        std::ifstream in(path); std::string value;
        if (!(in >> value) || value == "max") return {};
        int64_t parsed = 0;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
        // A malformed readable limit must not become an unlimited budget.
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return 0;
        return std::max<int64_t>(0, parsed);
    };
    for (;;) {
        if (const auto limit = number(current / "memory.max")) {
            const auto usage = number(current / "memory.current");
            const auto used = usage.value_or(*limit);
            int64_t inactive = 0; std::ifstream stats(current / "memory.stat");
            for (std::string key; stats >> key;) {
                int64_t value; if (!(stats >> value)) break;
                if (usage && key == "inactive_file") inactive = std::max<int64_t>(0, value);
            }
            const auto charged = used - std::min(used, inactive);
            ram.total_mib = std::min(ram.total_mib, *limit / 1048576);
            ram.available_mib = std::min(ram.available_mib, std::max<int64_t>(0, *limit - charged) / 1048576);
        }
        if (const auto limit = number(current / "memory.swap.max")) {
            const auto used = number(current / "memory.swap.current").value_or(*limit);
            ram.swap_available_mib = std::min(ram.swap_available_mib, std::max<int64_t>(0, *limit - used) / 1048576);
        }
        if (current == base || current.empty()) break;
        current = current.parent_path();
    }
    ram.available_mib = std::min(ram.available_mib, ram.total_mib);
    return ram;
}
PromptCacheBudget native_llm_prompt_cache_budget(const std::optional<std::string> &raw, int64_t context,
    const LlmCacheEnvironment &environment, const std::function<PromptCacheRam()> &memory) {
    auto requested = resolve_llm_prompt_cache(raw, context);
    const auto unsafe = setting(environment("LLM_PROMPT_CACHE_ALLOW_UNSAFE", "false"));
    if (!configuration_integer_less("0", requested.mib) || unsafe == "1" || unsafe == "true" || unsafe == "yes" || unsafe == "on")
        return {requested, cap_llm_prompt_cache(requested, environment, memory)};
    PromptCacheRam ram{};
    try { ram = memory(); }
    catch (const LlmCacheError &error) {
        // Preserve the source policy's conservative fallback/error categories
        // when a memory probe cannot supply a trustworthy native extension.
        return {requested, cap_llm_prompt_cache(requested, environment,
            [&]() -> PromptCacheRam { throw error; })};
    }
    const auto automatic = setting(raw.value_or("auto"));
    if ((automatic.empty() || automatic == "auto") && ram.total_mib >= 65536) {
        requested.mib = std::to_string(std::max(std::stoll(requested.mib),
            static_cast<long long>(std::min<int64_t>(32768, ram.total_mib / 4))));
        requested.reason += "; high-RAM automatic ceiling";
    }
    const auto nonnegative = [&](const char *name, const char *fallback) {
        auto value = parse_configuration_integer(environment(name, fallback)).value_or(fallback);
        if (configuration_integer_less(value, "0")) return int64_t(0);
        if (configuration_integer_less(std::to_string(std::numeric_limits<int64_t>::max()), value)) return std::numeric_limits<int64_t>::max();
        return static_cast<int64_t>(std::stoll(value));
    };
    const auto swap_limit = nonnegative("LLM_PROMPT_CACHE_SWAP_MIB", "0");
    const auto swap_margin = nonnegative("LLM_PROMPT_CACHE_SWAP_MARGIN_MIB", "1024");
    auto swap = std::min(swap_limit, std::max<int64_t>(0, ram.swap_available_mib - swap_margin));
    if ((automatic.empty() || automatic == "auto") && swap > 0) {
        requested.mib = std::to_string(std::stoll(requested.mib) + swap);
    }
    auto effective = cap_llm_prompt_cache(requested, environment, [&] { return ram; });
    // Swap supplements a viable RAM budget; it cannot rescue an exhausted
    // container. The engine still needs resident memory to process requests.
    if (effective.mib == "0") swap = 0;
    if (swap > 0) {
        const auto original = std::stoll(effective.mib);
        effective.mib = minimum(requested.mib, std::to_string(original + swap));
        swap = std::stoll(effective.mib) - original;
        effective.reason += "; bounded existing swap allowance=" + std::to_string(swap) + " MiB";
    }
    return {requested, effective, swap};
}
PromptCacheRam native_prompt_cache_ram() {
    std::ifstream input("/proc/meminfo");
    if (!input) throw LlmCacheError("OSError", "Cannot read /proc/meminfo");
    const std::string meminfo{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    auto ram = prompt_cache_ram_from_proc(meminfo, []() -> std::optional<std::string> {
        std::ifstream input("/proc/zoneinfo"); if (!input) return {};
        return std::string{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }, sysconf(_SC_PAGESIZE));
    std::ifstream cgroups("/proc/self/cgroup");
    for (std::string line; std::getline(cgroups, line);) {
        if (line.compare(0, 3, "0::") == 0)
            return prompt_cache_ram_with_cgroup(ram, "/sys/fs/cgroup", line.substr(3));
    }
    // No unified controller is visible: retain the RAM policy, but do not
    // authorize host swap without knowing the container's swap allowance.
    ram.swap_available_mib = 0;
    return ram;
}
LlmCacheEnvironment native_llm_cache_environment() {
    return [](const std::string &name, const std::string &fallback) { const auto value = std::getenv(name.c_str()); return value ? std::string(value) : fallback; };
}
}
