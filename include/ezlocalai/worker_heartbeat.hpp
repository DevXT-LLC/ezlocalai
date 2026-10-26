#pragma once
#include <functional>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ezlocalai {
struct WorkerHeartbeatCancelled : std::runtime_error {
    WorkerHeartbeatCancelled() : std::runtime_error("Heartbeat cancelled during shutdown") {}
};
struct WorkerHeartbeatOperations {
    std::function<bool()> stopped;
    std::function<bool(const std::string &)> post;
    std::function<void(double)> wait;
    // Only successful steady-state registration may react to state changes;
    // registration/recovery backoff continues to use wait().
    std::function<void(double)> wait_for_update;
};

// Requests only wake the heartbeat owner; state collection never runs under a
// caller's admission, queue or resource lock. A fixed minimum spacing coalesces
// bursts without delaying a pending update indefinitely.
class WorkerHeartbeatSignal {
public:
    void start();
    void stop();
    void notify();
    void begin_update();
    void wait(double seconds, bool accept_updates);
private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool stopped_ = true;
    bool pending_ = false;
    std::chrono::steady_clock::time_point next_update_{};
};
// Initial registration backs off; steady-state recovery remains on the
// captured heartbeat interval, including after failed re-registration.
void run_worker_heartbeat(double interval, const WorkerHeartbeatOperations &operations);
}
