#include "ezlocalai/llm_stream.hpp"
#include <atomic>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {
void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool open = false;
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; cv.notify_all(); }
    void wait() { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return open; }); }
};
struct Unblock { Gate &gate; ~Unblock() { gate.release(); } };
const std::string first = R"({"choices":[{"delta":{"content":"first"},"finish_reason":null}]})";
const std::string second = R"({"choices":[{"delta":{"content":"second"},"finish_reason":null}]})";
const std::string final = R"({"choices":[{"delta":{},"finish_reason":"length"}]})";

void completion_releases_admission() {
    Gate finish, admitted;
    std::atomic<int> starts{0}, releases{0}, followers{0};
    ezlocalai::LlmInferenceStream stream("completion-test", "none", "test", true,
        [&](const auto &emit, const auto &) { ++starts; emit(first); finish.wait(); },
        [&] { ++releases; admitted.release(); });
    Unblock unblock_finish{finish};
    std::string chunk;
    check(stream.next(&chunk) && chunk.find("first") != std::string::npos, "initial callback precedes completion");
    check(starts == 1 && releases == 0, "active stream retains admission");
    auto follower = std::async(std::launch::async, [&] { admitted.wait(); ++followers; });
    auto drain = std::async(std::launch::async, [&] {
        std::string value;
        check(stream.next(&value) && value.find("\"finish_reason\": \"stop\"") != std::string::npos,
            "completion without a final callback supplies one stop event");
        check(releases == 0, "admission remains held while the final event is observable");
        check(!stream.next(&value), "stream exhausts after the final event");
    });
    Unblock unblock_follower{admitted};
    finish.release(); drain.get(); follower.get();
    check(releases == 1 && followers == 1, "exhaustion admits the next request exactly once");
    stream.close(); stream.close();
    check(releases == 1, "closing an exhausted stream does not double-release admission");
}
void queued_callbacks_before_completion() {
    std::atomic<int> releases{0};
    ezlocalai::LlmInferenceStream stream("completion-test", "none", "test", true,
        [&](const auto &emit, const auto &) { emit(first); emit(second); emit(final); }, [&] { ++releases; });
    std::string chunk;
    check(stream.next(&chunk) && chunk.find("first") != std::string::npos, "first callback order");
    check(stream.next(&chunk) && chunk.find("second") != std::string::npos, "second callback order");
    check(stream.next(&chunk) && chunk.find("\"finish_reason\": \"length\"") != std::string::npos,
        "queued final reason is preserved");
    check(releases == 0, "queued callbacks drain before admission release");
    check(!stream.next(&chunk) && releases == 1 && !stream.failure(), "successful terminal state");
}
void queued_callbacks_before_error() {
    std::atomic<int> releases{0};
    ezlocalai::LlmInferenceStream stream("completion-test", "none", "test", true,
        [&](const auto &emit, const auto &) { emit(first); emit(second); throw std::runtime_error("controlled failure"); },
        [&] { ++releases; });
    std::string chunk;
    check(stream.next(&chunk) && chunk.find("first") != std::string::npos, "first callback survives later error");
    check(stream.next(&chunk) && chunk.find("second") != std::string::npos, "second callback survives later error");
    check(!stream.next(&chunk), "error does not produce a successful finish event");
    check(stream.failure() && stream.failure()->message == "controlled failure" && releases == 1,
        "error and release remain observable after queued callbacks");
}
void empty_completion() {
    std::atomic<int> releases{0};
    ezlocalai::LlmInferenceStream stream("completion-test", "none", "test", true,
        [](const auto &, const auto &) {}, [&] { ++releases; });
    std::string chunk;
    check(stream.next(&chunk) && chunk.find("empty_stream") != std::string::npos,
        "empty completion preserves its diagnostic event");
    check(releases == 0 && !stream.next(&chunk) && releases == 1, "empty stream final event precedes release");
}
void abort_waiting_producer() {
    Gate finish;
    std::atomic<int> releases{0};
    std::atomic<bool> stopped{false};
    ezlocalai::LlmInferenceStream stream("completion-test", "none", "test", true,
        [&](const auto &emit, const auto &interrupt) {
            interrupt([&] { finish.release(); });
            emit(first); finish.wait(); stopped = emit(second);
        }, [&] { ++releases; });
    Unblock unblock_finish{finish};
    std::string chunk;
    check(stream.next(&chunk), "producer starts before abort");
    stream.abort_and_wait();
    check(stopped && releases == 1 && !stream.next(&chunk), "abort interrupts producer and releases once without late callbacks");
}
void close_unstarted_stream() {
    std::atomic<int> starts{0}, releases{0};
    ezlocalai::LlmInferenceStream stream("completion-test", "none", "test", true,
        [&](const auto &, const auto &) { ++starts; }, [&] { ++releases; });
    stream.close(); stream.close();
    check(starts == 0 && releases == 1, "unstarted close does not begin generation or double-release");
}
}
int main() {
    try {
        completion_releases_admission(); queued_callbacks_before_completion(); queued_callbacks_before_error();
        empty_completion(); abort_waiting_producer(); close_unstarted_stream();
        std::cout << "Stream completion, callback/final/error ordering, admission and cancellation checks passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
