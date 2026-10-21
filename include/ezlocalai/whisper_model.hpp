#pragma once
#include "ezlocalai/hub_snapshot.hpp"
#include "ezlocalai/whisper_transcription.hpp"
#include <functional>
namespace ezlocalai {
using WhisperSnapshotDownload = std::function<std::string(const std::string &,const HubSnapshotOptions &)>;
using WhisperTokenizerDownload = std::function<std::string(const std::string &,const std::string &,const HubSnapshotOptions &)>;
std::string resolve_whisper_tokenizer(const std::string &model_directory,bool multilingual,
    const WhisperTokenizerDownload &download = {});
std::string download_whisper_model(const std::string &name,const std::string &cache_directory,
    WhisperTranscriptionProfile profile,const WhisperSnapshotDownload &download = {});
std::string resolve_whisper_model(const std::string &name,const std::string &cache_directory,
    WhisperTranscriptionProfile profile);
// Source constructor filtering and preprocessor defaults. Missing/invalid JSON
// configurations use defaults; file decoding and non-object errors propagate.
std::string whisper_feature_kwargs(const std::string &model_directory);
WhisperFeatureConfiguration whisper_feature_configuration(const std::string &kwargs);
}
