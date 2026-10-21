#pragma once
#include <torch/nn/module.h>
#include <nlohmann/json.hpp>
namespace ezlocalai {
void save_wakeword_checkpoint(const std::string &directory, torch::nn::Module &module, const nlohmann::json &config);
void export_wakeword_onnx(const std::string &directory, torch::nn::Module &module, const nlohmann::json &config);
}
