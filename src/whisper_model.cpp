#include "ezlocalai/whisper_model.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <nlohmann/json.hpp>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
namespace ezlocalai {
namespace {
void validate_utf8(const std::string &text) {
    for (size_t at = 0; at < text.size();) {
        const auto first = static_cast<unsigned char>(text[at]); if (first < 128) { ++at; continue; }
        const size_t count = first >= 0xc2 && first <= 0xdf ? 2 : first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
        size_t end = at + 1; std::string reason = "invalid start byte";
        bool valid = count != 0;
        if (count) {
            for (size_t i = 1; i < count; ++i) {
                if (at + i == text.size()) { valid = false; end = text.size(); reason = "unexpected end of data"; break; }
                const auto byte = static_cast<unsigned char>(text[at + i]);
                const auto low = i == 1 && first == 0xe0 ? 0xa0 : i == 1 && first == 0xf0 ? 0x90 : 0x80;
                const auto high = i == 1 && first == 0xed ? 0x9f : i == 1 && first == 0xf4 ? 0x8f : 0xbf;
                if (byte < low || byte > high) { valid = false; end = at + i; reason = "invalid continuation byte"; break; }
            }
        }
        if (valid) { at += count; continue; }
        std::ostringstream message; message << "'utf-8' codec can't decode ";
        if (end == at + 1) message << "byte 0x" << std::hex << std::setw(2) << std::setfill('0') << unsigned(first) << std::dec << " in position " << at;
        else message << "bytes in position " << at << '-' << end - 1;
        message << ": " << reason; throw SttError("UnicodeDecodeError",message.str());
    }
}
}
std::string whisper_feature_kwargs(const std::string &directory) {
    using Json = nlohmann::ordered_json;
    const auto path = (std::filesystem::path(directory)/"preprocessor_config.json").string(); std::error_code ignored;
    if (path.find('\0') != std::string::npos || !std::filesystem::is_regular_file(path,ignored)) return "{}";
    std::ifstream input(path,std::ios::binary);
    if (!input) { const auto error = errno; throw SttError(error == EACCES ? "PermissionError" : error == ENOENT ? "FileNotFoundError" : "OSError","[Errno " + std::to_string(error) + "] " + std::strerror(error) + ": " + python_string_repr(path)); }
    const std::string text{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()}; validate_utf8(text);
    if (text.compare(0,3,"\xef\xbb\xbf") == 0) return "{}";
    Json value;
    try { value = Json::parse(text); } catch (const Json::parse_error &) { return "{}"; }
    if (!value.is_object()) {
        const auto type = value.is_null() ? "NoneType" : value.is_array() ? "list" : value.is_string() ? "str" : value.is_boolean() ? "bool" : value.is_number_integer() ? "int" : "float";
        throw SttError("AttributeError",std::string("'") + type + "' object has no attribute 'items'");
    }
    Json result = Json::object();
    for (const auto &item : value.items()) if (item.key() == "self" || item.key() == "feature_size" || item.key() == "sampling_rate" || item.key() == "hop_length" || item.key() == "chunk_length" || item.key() == "n_fft") result[item.key()] = item.value();
    return result.dump();
}
WhisperFeatureConfiguration whisper_feature_configuration(const std::string &kwargs) {
    using Json = nlohmann::json; const auto values = Json::parse(kwargs); WhisperFeatureConfiguration result;
    if (values.contains("self")) throw SttError("TypeError","FeatureExtractor.__init__() got multiple values for argument 'self'");
    for (auto entry : {std::pair{"feature_size",&result.feature_size},{"sampling_rate",&result.sampling_rate},{"hop_length",&result.hop_length},{"chunk_length",&result.chunk_length},{"n_fft",&result.n_fft}}) {
        if (!values.contains(entry.first)) continue; const auto &value = values.at(entry.first);
        if (value.is_boolean()) *entry.second = value.get<bool>();
        else if (value.is_string() && std::string(entry.first) == "feature_size") {
            const auto number = parse_configuration_integer(value.get<std::string>());
            if (!number) throw SttError("ValueError","invalid literal for int() with base 10: " + python_string_repr(value.get<std::string>()));
            *entry.second = std::stoll(*number);
        } else *entry.second = value.get<int64_t>();
    }
    if (!result.hop_length) throw SttError("ZeroDivisionError","integer division or modulo by zero");
    if (!result.sampling_rate) throw SttError("ZeroDivisionError","division by zero");
    if (!result.n_fft) throw SttError("ZeroDivisionError","float division by zero");
    return result;
}
std::string download_whisper_model(const std::string &name,const std::string &cache_directory,WhisperTranscriptionProfile profile,const WhisperSnapshotDownload &download) {
    std::vector<std::pair<std::string,std::string>> models;
    for (const auto *alias : {"tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large-v3"})
        models.emplace_back(alias,std::string("Systran/faster-whisper-") + alias);
    models.emplace_back("large","Systran/faster-whisper-large-v3");
    for (const auto *alias : {"large-v2","medium.en","small.en","large-v3"}) models.emplace_back(std::string("distil-") + alias,std::string("Systran/faster-distil-whisper-") + alias);
    if (profile == WhisperTranscriptionProfile::faster_whisper_1_2_0) {
        models.emplace_back("distil-large-v3.5","distil-whisper/distil-large-v3.5-ct2");
        for (const auto *alias : {"large-v3-turbo","turbo"}) models.emplace_back(alias,"mobiuslabsgmbh/faster-whisper-large-v3-turbo");
    }
    std::string repository; const auto slash = name.find('/'), newline = name.find('\n');
    if (slash != std::string::npos && slash < newline) repository = name;
    else {
        for (const auto &model : models) if (model.first == name) { repository = model.second; break; }
        if (repository.empty()) {
            std::string expected; for (const auto &model : models) { if (!expected.empty()) expected += ", "; expected += model.first; }
            throw SttError("ValueError","Invalid model size '" + name + "', expected one of: " + expected);
        }
    }
    HubSnapshotOptions options; options.cache_directory = cache_directory;
    options.allow_patterns = {"config.json","preprocessor_config.json","model.bin","tokenizer.json","vocabulary.*"};
    const auto call = [&] { return download ? download(repository,options) : hub_snapshot_download(repository,options); };
    try { return call(); } catch (const HubError &error) {
        if (!error.http && error.category != "ConnectionError") throw;
        options.local_files_only = true; return call();
    }
}
std::string resolve_whisper_model(const std::string &name,const std::string &cache_directory,WhisperTranscriptionProfile profile) {
    std::error_code ignored;
    if (name.find('\0') == std::string::npos && std::filesystem::is_directory(name,ignored)) return name;
    try { return download_whisper_model(name,cache_directory,profile); }
    catch (const HubError &error) { throw SttError(error.category,error.what(),error.http || error.category == "LocalEntryNotFoundError" || error.category == "OSError"); }
}
std::string resolve_whisper_tokenizer(const std::string &directory,bool multilingual,const WhisperTokenizerDownload &download) {
    const auto path = (std::filesystem::path(directory)/"tokenizer.json").string(); std::error_code ignored;
    if (path.find('\0') == std::string::npos && std::filesystem::is_regular_file(path,ignored)) return path;
    const std::string repository = multilingual ? "openai/whisper-tiny" : "openai/whisper-tiny.en";
    // Tokenizer.from_pretrained uses hf_hub_download's default cache, not the
    // model constructor's download_root, and propagates download errors.
    try { return download ? download(repository,"tokenizer.json",{}) : hub_snapshot_file(repository,"tokenizer.json",{}); }
    catch (const HubError &error) { throw SttError(error.category,error.what(),error.http || error.category == "LocalEntryNotFoundError" || error.category == "OSError"); }
}
}
