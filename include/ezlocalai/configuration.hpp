#pragma once

#include <string>
#include <optional>
#include <utility>
#include <vector>

namespace ezlocalai {

// Python str.strip() whitespace, including the Unicode whitespace recognized
// by the reference's environment and capability configuration.
std::string strip_config_value(const std::string &value);
std::string normalize_decimal_digits(const std::string &value);
// Python decimal conversion at the reference runtime's default digit limit.
// Successful results are canonical signed decimal strings, without range loss.
std::optional<std::string> parse_configuration_integer(const std::string &value);
bool configuration_integer_less(const std::string &left, const std::string &right);
bool parse_configuration_float(const std::string &value, double *out);
// Pipes service clients preserve whitespace and sentinel-looking URL values.
// Only the exact case-insensitive word "true" selects dedicated-server mode.
struct ServiceClientSettings {
    std::string raw, url, api_key;
    bool dedicated = false;
};
ServiceClientSettings capture_service_client_settings(const std::string &name,
    const std::string &base_url = "", const std::string &api_key = "");
std::vector<std::string> local_capabilities();
std::vector<std::pair<std::string, std::string>> capability_model_names(const std::vector<std::string> &capabilities);

struct EmbeddingModelConfig {
    std::string model;
    std::string alias;
    std::string quant;
    std::string path;
    std::string hf_argument;
    bool acquire = true;
};
EmbeddingModelConfig embedding_model_config();

} // namespace ezlocalai
