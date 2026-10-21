#pragma once
#include <functional>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace ezlocalai {
struct CttsCacheOperations {
    // Naive local datetime.isoformat(), evaluated separately at each source call.
    std::function<std::string()> now;
    std::function<int64_t(const std::string &)> duration_ms;
};
struct CttsCacheSelection { std::string audio,metadata; };
using CttsCacheGenerator=std::function<std::string(const std::string &)>;
std::string ctts_cache_key(const std::string &text,const std::string &voice,const std::string &language="en",const std::string &extra_json="null");
int64_t ctts_cache_audio_duration(const std::string &path);
class CttsAudioCache {
public:
    explicit CttsAudioCache(const std::string &configuration_json="null",CttsCacheOperations operations={});
    ~CttsAudioCache();
    CttsAudioCache(const CttsAudioCache &)=delete;
    CttsAudioCache &operator=(const CttsAudioCache &)=delete;
    std::optional<std::string> get(const std::string &key);
    // Source store mutates the caller's metadata even if a later write fails.
    bool store(const std::string &key,const std::string &audio,std::string &metadata_json);
    CttsCacheSelection select_best_audio(const CttsCacheGenerator &generate,const std::string &parameters_json);
    void warm(const std::vector<std::string> &phrases,const std::string &voice,const std::string &language,const CttsCacheGenerator &generate,const std::string &base_parameters_json);
    void cleanup_expired();
    void check_size();
    void remove(const std::string &key);
    void clear(const std::optional<std::string> &voice={});
    std::string statistics() const;
    std::string raw_statistics() const;
    std::string configuration() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
