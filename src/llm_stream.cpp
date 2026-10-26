#include "ezlocalai/llm_stream.hpp"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace ezlocalai {
namespace {
double wall_time() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
}
struct LlmInferenceStream::State {
    std::mutex mutex;
    std::condition_variable cv;
    std::string model, backend, version;
    Producer producer;
    std::function<void()> interrupt;
    std::vector<std::function<void()>> releases;
    std::deque<std::string> callbacks;
    std::optional<LlmResponseError> error;
    bool single_slot = true, started = false, done = false, exited = false;
    bool closed = false, released = false, aborted = false;

    void release() {
        std::vector<std::function<void()>> callbacks_to_run;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (released) return;
            released = true;
            callbacks_to_run.swap(releases);
        }
        for (auto &callback : callbacks_to_run) if (callback) callback();
    }
};

LlmInferenceStream::LlmInferenceStream(std::string model, std::string backend, std::string version,
    bool single_slot, Producer producer, std::function<void()> release) : state_(std::make_shared<State>()) {
    state_->model = std::move(model); state_->backend = std::move(backend); state_->version = std::move(version);
    state_->single_slot = single_slot; state_->producer = std::move(producer);
    state_->releases.push_back(std::move(release));
}
LlmInferenceStream::~LlmInferenceStream() { close(); }

void LlmInferenceStream::on_close(std::function<void()> release) {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->released) { state_->releases.push_back(std::move(release)); return; }
    }
    if (release) release();
}

bool LlmInferenceStream::next(std::string *chunk, const std::function<bool()> &disconnected) {
    if (disconnected && disconnected()) { close(); return false; }
    const auto state = state_;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->closed) return false;
        if (!state->started) {
            state->started = true;
            std::thread([state] {
                try {
                    state->producer([state](const std::string &value) {
                        {
                            std::lock_guard<std::mutex> lock(state->mutex);
                            if (state->closed) return true;
                            if (auto failure = llm_response_error(value, true)) { state->error = std::move(failure); return true; }
                        }
                        // Batch members enter the queue individually, without
                        // another callback error/cancellation check per member.
                        for (auto &item : llm_stream_callback_items(value)) {
                            std::lock_guard<std::mutex> lock(state->mutex);
                            state->callbacks.push_back(std::move(item)); state->cv.notify_all();
                        }
                        return false;
                    }, [state](std::function<void()> interrupt) {
                        bool aborted;
                        { std::lock_guard<std::mutex> lock(state->mutex); state->interrupt = interrupt; aborted = state->aborted; }
                        if (aborted && interrupt) interrupt();
                    });
                } catch (const std::exception &error) {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->error = LlmResponseError{"RuntimeError", error.what()};
                }
                bool closed;
                { std::lock_guard<std::mutex> lock(state->mutex); state->done = true; state->interrupt = {}; closed = state->closed; }
                if (closed) state->release();
                { std::lock_guard<std::mutex> lock(state->mutex); state->exited = true; state->cv.notify_all(); }
            }).detach();
            // The source starts native generation before assigning these two
            // separately sampled timestamps.
            const auto id_time = static_cast<int64_t>(wall_time()), created = static_cast<int64_t>(wall_time());
            normalizer_ = std::make_unique<LlmChatStreamNormalizer>(state->model, state->backend, state->version, id_time, created);
            last_keepalive_ = wall_time();
        }
    }
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->closed) return false;
        }
        if (!pending_.empty()) {
            // Starlette's synchronous iterator runs in AnyIO's worker thread.
            // A disconnect does not abandon its pending next(); cleanup begins
            // only once that call yields (or raises/exhausts).
            if (disconnected && disconnected()) { close(); return false; }
            *chunk = std::move(pending_.front()); pending_.pop_front(); return true;
        }
        if (exhausted_) { close(); return false; }
        std::string value;
        bool done = false, timed_out = false;
        std::optional<LlmResponseError> error;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            if (state->callbacks.empty() && !state->done && !state->closed) {
                // Native completion can finalize the stream and release its
                // admission lease immediately. Queued callbacks still take
                // precedence below, including callbacks before an error.
                timed_out = !state->cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return !state->callbacks.empty() || state->done || state->closed;
                });
            }
            if (state->closed) return false;
            if (!state->callbacks.empty()) { value = std::move(state->callbacks.front()); state->callbacks.pop_front(); }
            else { done = state->done; error = state->error; }
        }
        if (!value.empty()) {
            if (passthrough_) pending_.push_back(std::move(value));
            else for (auto &item : normalizer_->queued_chunk(value)) pending_.push_back(std::move(item));
        } else if (timed_out && wall_time() - last_keepalive_ >= 5.0) {
            last_keepalive_ = wall_time();
            if (disconnected && disconnected()) { close(); return false; }
            *chunk = normalizer_->keepalive(); return true;
        } else if (done) {
            exhausted_ = true;
            if (error) { close(); return false; }
            if (!passthrough_) for (auto &item : normalizer_->finish()) pending_.push_back(std::move(item));
        }
    }
}

void LlmInferenceStream::close() {
    const auto state = state_;
    bool release;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        if (state->closed) return;
        state->closed = true; state->callbacks.clear(); state->cv.notify_all();
        if (state->started && !state->exited) state->cv.wait_for(lock, std::chrono::seconds(5), [&] { return state->exited; });
        release = !state->started || state->exited || !state->single_slot;
    }
    if (release) state->release();
}

void LlmInferenceStream::abort_and_wait() {
    abort_state(state_);
}

LlmInferenceStream::OwnerHandle LlmInferenceStream::owner_handle() const {
    const std::weak_ptr<State> weak = state_;
    return {weak, [weak] { if (auto state = weak.lock()) abort_state(state); }};
}

void LlmInferenceStream::abort_state(const std::shared_ptr<State> &state) {
    std::function<void()> interrupt;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->closed = state->aborted = true; state->callbacks.clear(); interrupt = state->interrupt; state->cv.notify_all();
    }
    if (interrupt) interrupt();
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cv.wait(lock, [&] { return !state->started || state->exited; });
    }
    state->release();
}

std::optional<LlmResponseError> LlmInferenceStream::failure() const {
    std::lock_guard<std::mutex> lock(state_->mutex); return state_->error;
}

InferenceResult collect_llm_stream(InferenceResult result) {
    if (!result.stream) return result;
    std::string chunk, prefix;
    while (result.stream->next(&chunk)) prefix += "data: " + chunk + "\n\n";
    result.failure = result.stream->failure();
    result.lazy_chat_stream = result.failure.has_value();
    if (result.failure) { result.stream_prefix = prefix; result.body = prefix + llm_stream_error_event(result.failure->message); }
    else result.body = prefix + "data: [DONE]\n\n";
    result.stream.reset(); return result;
}
}
