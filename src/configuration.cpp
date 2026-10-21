#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ezlocalai_c.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace ezlocalai {
namespace {
bool python_space(uint32_t codepoint) {
    return (codepoint >= 0x09 && codepoint <= 0x0d) ||
           (codepoint >= 0x1c && codepoint <= 0x20) || codepoint == 0x85 || codepoint == 0xa0 ||
           codepoint == 0x1680 || (codepoint >= 0x2000 && codepoint <= 0x200a) ||
           codepoint == 0x2028 || codepoint == 0x2029 || codepoint == 0x202f || codepoint == 0x205f || codepoint == 0x3000;
}

uint32_t next_codepoint(const std::string &value, size_t &position) {
    const auto first = static_cast<unsigned char>(value[position++]);
    if (first < 0x80) return first;
    const unsigned count = first >= 0xc2 && first <= 0xdf ? 1 : first >= 0xe0 && first <= 0xef ? 2 : first >= 0xf0 && first <= 0xf4 ? 3 : 0;
    if (!count || position + count > value.size()) return 0xffffffff;
    uint32_t codepoint = first & (0x7f >> count);
    for (unsigned i = 0; i < count; ++i) {
        const auto byte = static_cast<unsigned char>(value[position + i]);
        if ((byte & 0xc0) != 0x80) return 0xffffffff;
        codepoint = (codepoint << 6) | (byte & 0x3f);
    }
    if ((count == 1 && codepoint < 0x80) || (count == 2 && codepoint < 0x800) ||
        (count == 3 && codepoint < 0x10000) || codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) return 0xffffffff;
    position += count;
    return codepoint;
}

// Configuration predicates compare ASCII words/substrings. U+212A is the
// only non-ASCII scalar whose Python lowercase mapping is one ASCII letter.
std::string lower_for_ascii_match(std::string value) {
    for (char &ch : value) if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
    for (size_t offset = 0; (offset = value.find("\xe2\x84\xaa", offset)) != std::string::npos;) value.replace(offset, 3, "k");
    return value;
}

std::string config(const char *name, const char *empty_fallback = "", const char *explicit_default = nullptr) {
    const auto *raw = ezl_getenv_fallback(name, explicit_default);
    return lower_for_ascii_match(strip_config_value(*raw ? raw : empty_fallback));
}

bool url(const std::string &value) {
    return value.compare(0, 7, "http://") == 0 || value.compare(0, 8, "https://") == 0;
}

bool has_projector(const std::string &model) {
    // Router.detect_local_capabilities uses the raw comma-separated string,
    // case-sensitive -GGUF removal, and a case-sensitive .gguf extension.
    const auto slash = model.rfind('/');
    auto basename = slash == std::string::npos ? model : model.substr(slash + 1);
    const auto suffix = basename.find("-GGUF");
    if (suffix != std::string::npos) basename.resize(suffix);
    std::error_code error;
    std::filesystem::directory_iterator it(std::filesystem::path("models") / basename, error), end;
    while (!error && it != end) {
        const auto name = it->path().filename().string();
        if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".gguf") == 0 &&
            lower_for_ascii_match(name).find("mmproj") != std::string::npos) return true;
        it.increment(error);
    }
    return false;
}
} // namespace

std::string strip_config_value(const std::string &value) {
    size_t first = value.size(), last = 0, offset = 0;
    while (offset < value.size()) {
        const size_t begin = offset;
        if (!python_space(next_codepoint(value, offset))) {
            first = std::min(first, begin);
            last = offset;
        }
    }
    return first == value.size() ? "" : value.substr(first, last - first);
}

ServiceClientSettings capture_service_client_settings(const std::string &name,
    const std::string &base_url, const std::string &api_key) {
    ServiceClientSettings client;
    client.raw = base_url.empty() ? ezl_getenv_default(name.c_str()) : base_url;
    client.dedicated = lower_for_ascii_match(client.raw) == "true";
    client.url = client.dedicated ? "" : client.raw;
    while (!client.url.empty() && client.url.back() == '/') client.url.pop_back();
    client.api_key = api_key.empty() ? ezl_getenv_default((name + "_API_KEY").c_str()) : api_key;
    if (client.api_key.empty()) client.api_key = ezl_getenv_default("EZLOCALAI_API_KEY");
    return client;
}

std::string normalize_decimal_digits(const std::string &value) {
    // Decimal (Nd) zero codepoints in the reference Python Unicode database.
    static constexpr uint32_t zeros[] = {0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, 0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0};
    std::string result;
    for (size_t offset = 0; offset < value.size();) {
        const auto begin = offset, codepoint = static_cast<size_t>(next_codepoint(value, offset));
        bool found = false;
        for (auto zero : zeros) if (codepoint >= zero && codepoint <= zero + 9) { result += static_cast<char>('0' + codepoint - zero); found = true; break; }
        if (!found) result.append(value, begin, offset - begin);
    }
    return result;
}

std::optional<std::string> parse_configuration_integer(const std::string &raw) {
    for (unsigned char c : raw) if (c >= 0x1c && c <= 0x1f) return {};
    const auto value = normalize_decimal_digits(strip_config_value(raw));
    size_t start = 0; bool negative = false, digit = false;
    if (!value.empty() && (value[0] == '+' || value[0] == '-')) { start = 1; negative = value[0] == '-'; }
    std::string digits;
    for (size_t i = start; i < value.size(); ++i) {
        const auto c = value[i];
        if (c >= '0' && c <= '9') { digit = true; digits += c; }
        else if (c == '_' && digit) digit = false;
        else return {};
    }
    if (!digit || digits.size() > 4300) return {};
    const auto first = digits.find_first_not_of('0');
    if (first == std::string::npos) return "0";
    digits.erase(0, first);
    return negative ? '-' + digits : digits;
}
bool configuration_integer_less(const std::string &left, const std::string &right) {
    const bool ln = left[0] == '-', rn = right[0] == '-';
    if (ln != rn) return ln;
    if (left.size() != right.size()) return ln ? left.size() > right.size() : left.size() < right.size();
    return ln ? left > right : left < right;
}
bool parse_configuration_float(const std::string &raw, double *out) {
    for (unsigned char c : raw) if (c >= 0x1c && c <= 0x1f) return false;
    const auto value = normalize_decimal_digits(strip_config_value(raw));
    if (value.empty()) return false;
    size_t i = 0;
    if (value[i] == '+' || value[i] == '-') ++i;
    auto special = value.substr(i);
    for (char &c : special) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    if (special != "nan" && special != "inf" && special != "infinity") {
        const auto digits = [&]() {
            bool any = false;
            while (i < value.size() && value[i] >= '0' && value[i] <= '9') {
                any = true; ++i;
                if (i < value.size() && value[i] == '_' && i + 1 < value.size() && value[i+1] >= '0' && value[i+1] <= '9') ++i;
            }
            return any;
        };
        bool any = digits();
        if (i < value.size() && value[i] == '.') { ++i; any = digits() || any; }
        if (!any) return false;
        if (i < value.size() && (value[i] == 'e' || value[i] == 'E')) {
            ++i; if (i < value.size() && (value[i] == '+' || value[i] == '-')) ++i;
            if (!digits()) return false;
        }
        if (i != value.size()) return false;
    }
    std::string normalized;
    for (char c : value) if (c != '_') normalized += c;
    char *end = nullptr; *out = std::strtod(normalized.c_str(), &end);
    return end == normalized.c_str() + normalized.size();
}

std::vector<std::string> local_capabilities() {
    std::vector<std::string> caps;
    const auto model_raw = strip_config_value(ezl_getenv_default("DEFAULT_MODEL"));
    const auto model = lower_for_ascii_match(model_raw);
    const auto voice_server = config("VOICE_SERVER");
    const auto image_server = config("IMAGE_SERVER");
    const auto text_server = config("TEXT_SERVER");
    const auto embedding_server = config("EMBEDDING_SERVER");
    const auto image_model = config("IMG_MODEL");
    const auto video_model = config("VIDEO_MODEL", "unsloth/LTX-2.3-GGUF", "unsloth/LTX-2.3-GGUF");
    const auto music_model = config("MUSIC_MODEL", "Serveurperso/ACE-Step-1.5-GGUF", "Serveurperso/ACE-Step-1.5-GGUF");
    const bool music_enabled = config("MUSIC_ENABLED", "false") == "true";
    if (!model.empty() && model != "none" && !url(text_server)) {
        caps.push_back("text");
        bool vision = false;
        for (const std::string_view tag : {"-vl", "-vlm", "vision", "qwen3.6", "qwen3.8", "qvq", "minicpm-v",
                                          "llava", "bakllava", "moondream", "cogvlm", "internvl", "idefics"}) {
            if (model.find(tag) != std::string::npos) { vision = true; break; }
        }
        if (vision || has_projector(model_raw)) caps.push_back("vision");
    }
    if (!url(voice_server) && config("TTS_ENABLED", "true") == "true") caps.push_back("tts");
    if (!url(voice_server) && config("STT_ENABLED", "true") == "true") caps.push_back("stt");
    if (config("IMAGE_ENABLED", "false") == "true" && !url(image_server) &&
        ((!image_model.empty() && image_model != "none") || image_server == "true")) caps.push_back("image");
    if (config("VIDEO_ENABLED", "false") == "true" && !url(image_server) && !video_model.empty() && video_model != "none") {
        caps.push_back("video");
        if (music_enabled && !music_model.empty() && music_model != "none") caps.push_back("music_video");
    }
    if (music_enabled && !music_model.empty() && music_model != "none") caps.push_back("music");
    if (config("EMBEDDING_ENABLED", "true") == "true" && !url(embedding_server)) caps.push_back("embedding");
    return caps;
}

std::vector<std::pair<std::string, std::string>> capability_model_names(const std::vector<std::string> &capabilities) {
    auto model_value = [](const char *name, const char *fallback = "", const char *explicit_default = nullptr) {
        const char *raw = ezl_getenv_fallback(name, explicit_default);
        return strip_config_value(*raw ? raw : fallback);
    };
    const auto video = model_value("VIDEO_MODEL", "unsloth/LTX-2.3-GGUF", "unsloth/LTX-2.3-GGUF");
    const auto music = model_value("MUSIC_MODEL", "Serveurperso/ACE-Step-1.5-GGUF", "Serveurperso/ACE-Step-1.5-GGUF");
    auto usable = [](const std::string &name) { return !name.empty() && lower_for_ascii_match(name) != "none"; };
    std::vector<std::pair<std::string, std::string>> models;
    for (const auto &cap : capabilities) {
        std::string model;
        if (cap == "tts") {
            model = strip_config_value(ezl_getenv_default("QWEN_TTS_MODEL"));
            if (model.empty() || model == "Qwen/Qwen3-TTS-12Hz-0.6B-Base") model = "Mouserat/qwen3-tts-0.6b-base-gguf";
        } else if (cap == "stt") model = model_value("WHISPER_MODEL", "large-v3-turbo");
        else if (cap == "image") model = model_value("IMG_MODEL");
        else if (cap == "video" && usable(video)) model = video;
        else if (cap == "music_video" && usable(video) && usable(music)) model = music + " + " + video;
        else if (cap == "music" && usable(music)) model = music;
        else if (cap == "embedding") {
            model = model_value("EMBEDDING_MODEL_ALIAS");
            if (model.empty()) model = model_value("EMBEDDING_MODEL");
        }
        // Reference always includes TTS and STT, even a whitespace-only STT
        // value, whereas optional capability model fields omit empty names.
        if (!model.empty() || cap == "tts" || cap == "stt") models.emplace_back(cap, model);
    }
    return models;
}

EmbeddingModelConfig embedding_model_config() {
    EmbeddingModelConfig result;
    result.model = strip_config_value(ezl_getenv_default("LLAMA_CPP_EMBEDDING_MODEL"));
    if (result.model.empty()) result.model = ezl_getenv_default("EMBEDDING_MODEL");
    result.alias = strip_config_value(ezl_getenv_default("LLAMA_CPP_EMBEDDING_ALIAS"));
    if (result.alias.empty()) result.alias = ezl_getenv_default("EMBEDDING_MODEL_ALIAS");
    if (result.alias.empty()) result.alias = result.model;
    result.quant = ezl_getenv_default("EMBEDDING_QUANT_TYPE");
    result.hf_argument = result.model;
    if (!result.quant.empty() && result.model.find(':') == std::string::npos) result.hf_argument += ":" + result.quant;
    result.path = strip_config_value(ezl_getenv_default("LLAMA_CPP_EMBEDDING_MODEL_PATH"));
    if (!result.path.empty()) { result.acquire = false; return result; }
    std::error_code error;
    if (std::filesystem::is_regular_file(result.model, error)) {
        result.path = result.model;
        result.acquire = false;
        return result;
    }
    auto basename = lower_for_ascii_match(std::filesystem::path(result.model).filename().string());
    const auto suffix = basename.find("-gguf");
    if (suffix != std::string::npos) basename.resize(suffix);
    if (basename.empty()) return result;
    const auto quant = lower_for_ascii_match(result.quant);
    // MODELS_PATH is an explicit native override. Default discovery follows
    // LLM.download_model's model-specific directory and case-sensitive quant.
    for (const auto &root : {std::string(ezl_getenv_default("MODELS_PATH"))}) {
        if (root.empty()) continue;
        error.clear();
        std::filesystem::recursive_directory_iterator it(root, std::filesystem::directory_options::skip_permission_denied, error), end;
        std::filesystem::path selected;
        bool selected_model_directory = false;
        while (!error && it != end) {
            const auto filename = lower_for_ascii_match(it->path().filename().string());
            const auto directory = lower_for_ascii_match(it->path().parent_path().filename().string());
            if (it->is_regular_file(error) && it->path().extension() == ".gguf" && filename.find("mmproj") == std::string::npos &&
                (filename.find(basename) != std::string::npos || directory == basename) &&
                (quant.empty() || filename.find(quant) != std::string::npos)) {
                // Other engines can cache a different checkpoint with the
                // same filename (ACE-Step's embedding GGUF is one example).
                // Prefer the model-specific directory used by download_model;
                // do not let filesystem iteration order select another copy.
                const bool model_directory = directory == basename;
                if (selected.empty() || (model_directory && !selected_model_directory)
                    || (model_directory == selected_model_directory && it->path() < selected)) {
                    selected = it->path();
                    selected_model_directory = model_directory;
                }
            }
            it.increment(error);
        }
        if (!selected.empty()) {
            result.path = std::filesystem::absolute(selected).string();
            result.acquire = false;
            return result;
        }
    }
    auto repository = result.model;
    if (repository.find('/') == std::string::npos) repository = "unsloth/" + repository + "-GGUF";
    auto model = repository.substr(repository.rfind('/') + 1);
    const auto model_suffix = model.find("-GGUF"); if (model_suffix != std::string::npos) model.resize(model_suffix);
    auto requested_quant = result.quant;
    const auto comma = requested_quant.find(','); if (comma != std::string::npos) requested_quant = strip_config_value(requested_quant.substr(0, comma));
    error.clear();
    std::filesystem::directory_iterator local(std::filesystem::path("models") / model, error), end;
    while (!error && local != end) {
        const auto filename = local->path().filename().string();
        if (filename.size() >= 5 && filename.compare(filename.size() - 5, 5, ".gguf") == 0 && lower_for_ascii_match(filename).find("mmproj") == std::string::npos && (requested_quant.empty() || filename.find(requested_quant) != std::string::npos)) {
            result.path = std::filesystem::absolute(local->path()).string(); return result;
        }
        local.increment(error);
    }
    return result;
}
} // namespace ezlocalai
