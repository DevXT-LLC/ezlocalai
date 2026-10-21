#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ezlocalai {
struct ModelFiles { std::string model, projector; };
// The policy is shared with LLM.download_model. Hooks allow its filesystem and
// Hub call sequence to be compared directly with the source oracle.
struct ModelDownloadHooks {
    std::function<void(const std::string &)> mkdir;
    std::function<std::vector<std::string>(const std::string &)> list_local;
    std::function<bool(const std::string &)> exists;
    std::function<std::vector<std::string>(const std::string &)> list_remote;
    std::function<void(const std::string &, const std::string &, const std::string &)> download;
};
ModelFiles download_gguf_model(std::string model_name, const std::string &models_dir = "models",
    std::optional<std::string> quant = std::nullopt, bool skip_projector = false,
    ModelDownloadHooks hooks = {});
}
