#include "ezlocalai/llm_stream.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
struct Gate {
    std::mutex mutex; std::condition_variable cv; bool open = false;
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; cv.notify_all(); }
    void wait() { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return open; }); }
};
double sample() {
    Gate finish;
    std::atomic<int> releases{0};
    std::chrono::steady_clock::time_point producer_finished, admission_released;
    ezlocalai::LlmInferenceStream stream("completion-benchmark", "none", "test", true,
        [&](const auto &emit, const auto &) {
            emit(R"({"choices":[{"delta":{"content":"first"},"finish_reason":null}]})");
            finish.wait(); producer_finished = std::chrono::steady_clock::now();
        }, [&] { admission_released = std::chrono::steady_clock::now(); ++releases; });
    struct Unblock { Gate &gate; ~Unblock() { gate.release(); } } unblock{finish};
    std::string chunk;
    check(stream.next(&chunk) && chunk.find("first") != std::string::npos, "first callback");
    std::promise<void> entering;
    auto consuming = std::async(std::launch::async, [&] {
        entering.set_value();
        std::string final;
        check(stream.next(&final) && final.find("\"finish_reason\": \"stop\"") != std::string::npos, "implicit final event");
        check(releases == 0 && !stream.next(&final), "final event precedes exhaustion and release");
    });
    entering.get_future().wait();
    // Allow the consumer to enter its condition-variable wait, then complete
    // without emitting another callback that would otherwise wake that wait.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    finish.release(); consuming.get();
    check(releases == 1 && !stream.failure(), "successful completion releases admission once");
    return std::chrono::duration<double, std::milli>(admission_released - producer_finished).count();
}
}
int main(int argc, char **argv) {
    try {
        const int count = argc == 2 ? std::stoi(argv[1]) : 31;
        check(count > 0 && count <= 1000, "usage: bench_stream_completion [1..1000 repeats]");
        std::vector<double> samples;
        for (int i = 0; i < count; ++i) samples.push_back(sample());
        auto sorted = samples; std::sort(sorted.begin(), sorted.end());
        std::cout << std::setprecision(9)
            << "{\"scope\":\"producer completion to stream exhaustion and admission release, not TTFT or decode\",\"median_ms\":"
            << sorted[sorted.size()/2] << ",\"samples_ms\":[";
        for (size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
        std::cout << "],\"final_and_release_order_validated\":true}\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
