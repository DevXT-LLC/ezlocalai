#pragma once
#include <optional>
#include <string>
namespace ezlocalai {
// CPython's non-strict POSIX realpath, including dangling links and cycles.
std::string ctts_real_path(const std::string &path);
std::string ctts_contained_file_path(const std::string &base,const std::optional<std::string> &stem,const std::string &extension);
std::optional<std::string> ctts_voice_audio_path(const std::string &voices,const std::optional<std::string> &voice);
std::optional<std::string> ctts_voice_ref_text(const std::string &voices,const std::optional<std::string> &audio_path);
}
