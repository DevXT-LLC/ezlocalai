#pragma once
#include "ezlocalai/llama_tts.hpp"
#include <optional>
#include <utility>
namespace ezlocalai {
struct CttsGenerated { std::vector<TtsAudio> audio; std::string rate_json="24000"; };
struct CttsBackend {
    virtual ~CttsBackend() = default;
    std::string generate_name="ezlocalai.LlamaTTS.LlamaTTS.generate_voice_clone";
    std::function<CttsGenerated(const std::string &)> generate;
    std::function<std::unique_ptr<TtsStreamIterator>(const std::string &)> stream;
    std::function<void()> close;
};
struct CttsOperations {
    std::function<std::optional<std::string>(const std::string &)> environment;
    std::function<bool()> cuda_available;
    std::function<std::pair<double,double>(const std::string &)> memory_info;
    std::function<double(const std::string &)> total_bytes,allocated_bytes;
    std::function<void()> collect,empty_cache,synchronize,ipc_collect;
    // Enter a per-thread inference context; the returned function exits it once.
    std::function<std::function<void(const std::string &)>()> inference_mode;
    std::function<std::shared_ptr<CttsBackend>(const std::string &,const std::string &)> load;
};
struct CttsSettings {
    std::string model_id,max_chunk_chars,stream_chunk_target_chars,stream_first_chunk_target_chars,stream_min_first_chunk_chars,stream_write_bytes,stream_flush_silence_ms;
    double stream_frame_drain_seconds=0;
    bool non_streaming_mode=true,allow_cpu_fallback=true;
    std::string generation_kwargs;
    std::string json() const;
};
// Settings and model-lifecycle portion of CTTS. Filesystem/cache/transport are
// composed separately; construction here stops at the source sample_rate=24000.
class CttsRuntime {
public:
    explicit CttsRuntime(const std::string &requested_device_json="null",CttsOperations operations={});
    ~CttsRuntime();
    CttsRuntime(const CttsRuntime &)=delete;
    CttsRuntime &operator=(const CttsRuntime &)=delete;
    TtsAudio generate_sample(const std::string &request_json);
    std::string warmup_generation_kwargs() const;
    void release_generation_memory();
    void reload_on_cpu();
    void close();
    const CttsSettings &settings() const;
    std::string device() const;
    bool closed() const;
    std::shared_ptr<CttsBackend> backend() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Supplies real native LlamaTts workers. Hardware hooks may be supplied by the
// optional LibTorch adapter; without hardware hooks this platform is CPU-only.
CttsOperations ctts_llama_operations(CttsOperations operations={});
}
