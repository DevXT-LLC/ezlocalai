#pragma once
#include "ezlocalai/ctts_runtime.hpp"
#include "ezlocalai/ctts_cache.hpp"
namespace ezlocalai {
struct CttsServiceOperations {
    CttsOperations runtime;
    CttsCacheOperations cache;
    std::function<std::string()> uuid_hex;
    std::function<void(double)> stream_sleep;
};
class CttsByteStream {
public:
    virtual ~CttsByteStream()=default;
    virtual std::optional<std::string> next()=0;
    virtual void close()=0;
    virtual void request_cancel()=0;
};
// CTTS construction and generation. The caller owns request scheduling;
// each synthesis chunk runs on a worker, while preparation/cache/output remain
// on the calling thread. Stream next/close have one consumer; request_cancel
// may run concurrently. Pool and HTTP scheduling compose separately.
class Ctts {
public:
    explicit Ctts(const std::string &cache_configuration_json="null",const std::string &device_json="null",CttsServiceOperations operations={});
    ~Ctts();
    Ctts(const Ctts &)=delete;
    Ctts &operator=(const Ctts &)=delete;
    std::string generate(const std::string &request_json);
    std::unique_ptr<CttsByteStream> generate_stream(const std::string &request_json);
    void close();
    std::string sample_rate() const;
    const std::vector<std::string> &voices() const;
    const std::string &output_folder() const;
    const std::string &voices_path() const;
    std::string use_cache_json() const;
    CttsRuntime &runtime();
    CttsAudioCache &cache();
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
}
