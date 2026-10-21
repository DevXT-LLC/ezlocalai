#pragma once
#include "ezlocalai/stt_pool.hpp"
#include "ezlocalai/voice_slots.hpp"
#include <atomic>
namespace ezlocalai {
class ResourceManager;
using SttPoolFactory = std::function<SttPoolOperations(std::shared_ptr<ResourceManager>)>;
struct SttRequestError : std::runtime_error { int status; SttRequestError(int status,std::string message) : std::runtime_error(std::move(message)),status(status) {} };
enum class SttServiceUse { Endpoint, LlmAudio, LlmVideo, TtsPipeline };
class SttService {
public:
    SttService(std::string model,int64_t capacity,SttPoolOperations pool,VoiceSlotOperations voice,
        std::shared_ptr<VoicePermitQueue> shared,std::shared_ptr<ResourceManager> resources,std::function<bool()> voice_server_mode);
    ~SttService();
    void warm();
    bool unload_idle();
    void cancel();
    // Inference ValueErrors become HTTP 400 only for the transcription endpoint.
    // LLM paths add their own resource mark. Audio destroys after success;
    // video destroys in finally, both irrespective of voice-server mode.
    std::string transcribe(const std::string &request,bool bad_request_on_value_error,uint64_t key = 0,SttServiceUse use = SttServiceUse::Endpoint);
    std::string transcribe_bytes(const std::string &request,const std::string &audio,uint64_t key=0);
    SttPoolSnapshot snapshot() const;
    bool handoff_active() const { return handoffs_.load() != 0; }
private:
    std::string transcribe_impl(const std::string &request,bool bad_request,uint64_t key,SttServiceUse use,const std::string *audio);
    std::shared_ptr<ResourceManager> resources_;
    std::function<bool()> voice_server_mode_;
    std::atomic<unsigned> handoffs_{0};
    VoiceCancellation cancellation_;
    std::unique_ptr<SttPool> pool_;
    std::unique_ptr<VoiceSlotGuard> guard_;
};
// Source app.py subtitle helpers; segments remain JSON to preserve fields.
std::string stt_segments_subtitles(const std::string &segments,bool vtt);
}
