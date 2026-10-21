#pragma once
#include "llm_response.hpp"
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace ezlocalai {
std::string llm_stream_delta(const std::string &json);
bool llm_stream_has_assistant_text(const std::string &json);
std::string llm_stream_finish_reason(const std::string &json);
std::vector<std::string> llm_stream_callback_items(const std::string &json);

// Values crossing the native callback boundary are JSON, including strings
// and batches. Returned chunks use app.py's json.dumps representation.
class LlmChatStreamNormalizer {
public:
    LlmChatStreamNormalizer(std::string model, std::string backend, std::string version,
        int64_t id_time, int64_t created);
    std::vector<std::string> callback(const std::string &json);
    std::vector<std::string> queued_chunk(const std::string &json);
    std::vector<std::string> finish();
    std::string keepalive() const;
    const std::optional<LlmResponseError> &failure() const { return failure_; }
private:
    std::string wrap(const std::string &delta, const std::string &reason) const;
    void chunk(const std::string &json, std::vector<std::string> *output);
    std::string model_, backend_, version_, id_;
    int64_t created_;
    size_t raw_chunks_ = 0, assistant_chunks_ = 0;
    bool finished_ = false;
    std::string pending_final_;
    std::map<std::string, std::string> diagnostics_;
    std::optional<LlmResponseError> failure_;
};

// A source-style lazy generator. The producer begins on the first next().
// Normal close signals cancellation at the next callback and joins for five
// seconds; a single-slot lease survives that timeout until native completion.
class LlmInferenceStream {
public:
    using Emit = std::function<bool(const std::string &)>;
    using Interrupt = std::function<void(std::function<void()>)>;
    using Producer = std::function<void(const Emit &, const Interrupt &)>;
    using OwnerHandle = std::pair<std::weak_ptr<void>, std::function<void()>>;
    LlmInferenceStream(std::string model, std::string backend, std::string version,
        bool single_slot, Producer producer, std::function<void()> release);
    ~LlmInferenceStream();
    bool next(std::string *chunk, const std::function<bool()> &disconnected = {});
    void close();
    void on_close(std::function<void()> release);
    // Owner shutdown, distinct from the source's bounded client cancellation.
    void abort_and_wait();
    OwnerHandle owner_handle() const;
    std::optional<LlmResponseError> failure() const;
    // Relay producer chunks unchanged. Used for chunks that another worker
    // already normalized, such as a router fallback stream.
    void set_passthrough(bool value) { passthrough_ = value; }
private:
    struct State;
    static void abort_state(const std::shared_ptr<State> &state);
    std::shared_ptr<State> state_;
    std::unique_ptr<LlmChatStreamNormalizer> normalizer_;
    std::deque<std::string> pending_;
    bool exhausted_ = false, passthrough_ = false;
    double last_keepalive_ = 0;
};

InferenceResult collect_llm_stream(InferenceResult result);
}
