#pragma once
#include "ezlocalai/tts_pool.hpp"
#include "ezlocalai/voice_slots.hpp"
namespace ezlocalai {
class TtsWebSocketLease;
// The caller supplies the fallback transport; this layer owns the source local
// request/lease lifetime and does not choose remote voice/router providers.
struct TtsRequestOperations {
    std::function<std::optional<std::string>(const std::string &)> environment;
    std::function<bool()> voice_server_mode;
    std::function<void(bool)> mark_in_use;
    std::function<std::string(const std::string &)> fallback_generate;
};
bool tts_should_try_gtts(const std::string &message,const std::string &configured);
class TtsServiceStream : public CttsByteStream {
public:
    // Explicit task identity; zero chooses one stable native consumer identity.
    // Set only from the stream's single consumer, before next/close.
    virtual void consumer_key(uint64_t key)=0;
};
class TtsService {
public:
    TtsService(int64_t capacity,TtsPoolOperations pool,VoiceSlotOperations voice,
        std::shared_ptr<VoicePermitQueue> shared,std::shared_ptr<ResourceManager> resources,TtsRequestOperations request);
    ~TtsService();
    void warm();
    void warm_cache();
    void unload(uint64_t key=0);
    bool unload_idle();
    // Cancels waiting/new admission. Active streams use request_cancel().
    void cancel();
    std::string generate(const std::string &request,uint64_t key=0);
    // Local Qwen only, with the same admission and cleanup as buffered speech.
    // Per-job cancellation can interrupt admission without cancelling the service.
    std::string generate_local(const std::string &request,const VoiceCancellation &cancellation={},uint64_t key=0);
    // Pipes PDF/audio pipelines return CTTS's URL/base64 string unchanged,
    // mark usage explicitly, and destroy only after successful generation.
    std::string generate_pipeline(const std::string &request,uint64_t key=0);
    // Acquires the slot and model now, before the caller commits HTTP headers.
    // Iteration transfers the lease and performs source aclosing/finally logic.
    std::unique_ptr<TtsServiceStream> stream(const std::string &request,uint64_t owner=0);
    // WebSocket synthesis retains one lease for the already-queued segments.
    std::unique_ptr<TtsWebSocketLease> websocket_lease(const VoiceCancellation &cancellation,uint64_t key=0);
    TtsPoolSnapshot snapshot() const;
    bool handoff_active() const;
    VoicePermitSnapshot permits() const;
    std::vector<std::pair<uint64_t,std::vector<VoiceSlotHandle>>> slot_stacks() const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
}
