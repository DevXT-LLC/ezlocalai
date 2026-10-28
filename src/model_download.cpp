#include "ezlocalai/model_download.hpp"
#include "ezlocalai/hub_assets.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ezlocalai_c.h"
#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/omni_model.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace ezlocalai {
namespace {
std::string lower(const std::string &value) { return llm_unicode_lower(value); }
bool gguf(const std::string &name) { return name.size() >= 5 && name.compare(name.size() - 5, 5, ".gguf") == 0; }
bool projector(const std::string &name) { return lower(name).find("mmproj") != std::string::npos; }
std::string join(const std::string &directory, const std::string &name) { return (std::filesystem::path(directory) / name).string(); }
}
ModelFiles download_gguf_model(std::string name, const std::string &models_dir, std::optional<std::string> quantization,
                               bool skip_projector, ModelDownloadHooks io) {
    if (!io.mkdir) io.mkdir = [](const auto &path) { std::filesystem::create_directories(path); };
    if (!io.list_local) io.list_local = [](const auto &path) {
        std::vector<std::string> names;
        for (const auto &entry : std::filesystem::directory_iterator(path)) names.push_back(entry.path().filename().string());
        return names;
    };
    if (!io.exists) io.exists = [](const auto &path) { return std::filesystem::exists(path); };
    if (!io.list_remote) io.list_remote = [](const auto &repo) { return list_hub_files(repo); };
    if (!io.download) io.download = [](const auto &repo, const auto &file, const auto &directory) { download_hub_file(repo, file, directory); };
    if (name.empty()) name = ezl_getenv_default("DEFAULT_MODEL");
    if (name.find('/') == std::string::npos) name = realtime_audio_model(name)
        ? "LiquidAI/" + name + (name.find("-GGUF") == std::string::npos ? "-GGUF" : "")
        : "unsloth/" + name + "-GGUF";
    std::string quant = quantization.value_or(ezl_getenv_default("QUANT_TYPE"));
    const auto comma = quant.find(',');
    if (comma != std::string::npos) quant = strip_config_value(quant.substr(0, comma));
    auto model = name.substr(name.rfind('/') + 1);
    const auto suffix = model.find("-GGUF"); if (suffix != std::string::npos) model.resize(suffix);
    const auto directory = join(models_dir, model);
    io.mkdir(directory);
    ModelFiles result;
    if (realtime_audio_model(name)) {
        // Resolve Liquid's model-qualified projector (encoders plus codec).
        // Known unsupported global Qwen quants fall back to full precision.
        auto desired = lower(quant);
        if (desired != "f16" && desired != "bf16" && desired != "q8_0" && desired != "q4_0" && desired != "q4_k_m") desired = "f16";
        const auto select = [&](const std::vector<std::string> &files, bool want_projector, const std::vector<std::string> &preferences) {
            for (const auto &preference : preferences) for (const auto &file : files)
                if (gguf(file) && projector(file) == want_projector && file.find('/') == std::string::npos
                    && lower(file).find(preference) != std::string::npos) return file;
            return std::string();
        };
        const auto local = io.list_local(directory);
        auto weight = select(local, false, {"-" + desired + ".gguf"});
        auto projection = select(local, true, {"-f16.", "-bf16.", "q8_0"});
        if (weight.empty() || (!skip_projector && projection.empty())) {
            const auto files = io.list_remote(name);
            if (weight.empty()) weight = select(files, false, {"-" + desired + ".gguf", "-f16.", "-bf16.", "q8_0", "q4_k_m"});
            if (!skip_projector && projection.empty()) projection = select(files, true, {"-f16.", "-bf16.", "q8_0"});
        }
        if (weight.empty() || (!skip_projector && projection.empty())) throw std::runtime_error("Missing LFM3 backbone or required audio/vision projector");
        result.model = join(directory, weight);
        if (!io.exists(result.model)) io.download(name, weight, directory);
        if (!skip_projector) {
            result.projector = join(directory, projection);
            if (!io.exists(result.projector)) io.download(name, projection, directory);
        }
        return result;
    }
    if (!skip_projector) {
        for (const auto &file : io.list_local(directory)) if (projector(file) && gguf(file)) { result.projector = join(directory, file); break; }
        if (result.projector.empty()) {
            const std::vector<std::string> candidates{"mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf", "mmproj-f16.gguf", "mmproj-model-f16.gguf", model + "-mmproj-f16.gguf", "mmproj.gguf", lower(model) + "-mmproj-f16.gguf"};
            for (const auto &file : candidates) {
                const auto path = join(directory, file);
                if (io.exists(path)) { result.projector = path; break; }
                try { io.download(name, file, directory); result.projector = path; break; }
                catch (const std::exception &) { /* Missing projectors are optional. */ }
            }
        }
    }
    std::vector<std::string> local;
    if (io.exists(directory)) for (const auto &file : io.list_local(directory)) if (gguf(file) && !projector(file)) local.push_back(file);
    for (const auto &file : local) if (quant.empty() || file.find(quant) != std::string::npos) { result.model = join(directory, file); return result; }
    try {
        std::vector<std::string> files;
        for (const auto &file : io.list_remote(name)) if (gguf(file) && !projector(file)) files.push_back(file);
        if (!files.empty()) {
            std::string chosen;
            for (const auto &pattern : {quant, std::string("Q4_K"), std::string("Q5_K"), std::string("Q6_K"), std::string("Q8")}) {
                for (const auto &file : files) if (file.find(pattern) != std::string::npos) { chosen = file; break; }
                if (!chosen.empty()) break;
            }
            if (chosen.empty()) chosen = files.front();
            io.download(name, chosen, directory);
            result.model = join(directory, chosen); return result;
        }
    } catch (const std::exception &error) { std::cerr << "[LLM] Failed to list/download from " << name << ": " << error.what() << '\n'; }
    throw std::runtime_error("No suitable model file found for " + name);
}
}
