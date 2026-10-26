#pragma once
#include "ezlocalai/ctts.hpp"
#include "ezlocalai/voice_slots.hpp"
#include <functional>

namespace ezlocalai {

// Validates one worker PCM segment, emitting the source's header/audio events.
// Each segment's zero-length terminator is consumed, not emitted.
class TtsPcmSegment {
public:
    using Emit = std::function<void(bool header,const std::string &bytes)>;
    void feed(const std::string &bytes,const Emit &emit);
    void finish() const;
private:
    std::string buffer_;
    bool header_=false,ended_=false;
    std::optional<uint32_t> remaining_;
};
struct TtsWebSocketInputResult {
    std::optional<std::string> request;
    bool done=false;
};
class TtsWebSocketInput {
public:
    // Input is JSON received from the client; output contains text/voice/language
    // keyword arguments for at most one queued synthesis segment.
    TtsWebSocketInputResult receive(const std::string &json);
private:
    std::u32string buffer_;
    std::string voice_="default",language_="en";
};

class TtsWebSocketLease {
public:
    virtual ~TtsWebSocketLease()=default;
    virtual std::unique_ptr<CttsByteStream> stream(const std::string &request)=0;
    virtual void close()=0;
};
struct TtsWebSocketOperations {
    // Transport callbacks enforce the source's per-receive/per-send timeout.
    std::function<std::string()> receive;
    std::function<void(const std::string &)> send;
    // Wakes receive after a synthesis failure without preventing the caller
    // from subsequently sending the WebSocket close frame.
    std::function<void()> interrupt_receive;
    std::function<std::unique_ptr<TtsWebSocketLease>(const VoiceCancellation &)> acquire;
    // Observes capacity after acquire/close, including cleanup failures. The
    // caller schedules state publication; observer errors do not affect audio.
    std::function<void()> state_changed;
};

// Receives concurrently with synthesis, queues at most four segments, and
// retains one local model lease while already-queued segments are drained.
// The caller maps failures to the app.py WebSocket close codes/reasons.
void run_tts_websocket(const TtsWebSocketOperations &operations);
std::string tts_pcm_header();

}
