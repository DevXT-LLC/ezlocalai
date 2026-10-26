#include "ezlocalai/worker_heartbeat.hpp"
#include <algorithm>
#include <cmath>

namespace ezlocalai {
void WorkerHeartbeatSignal::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = false;
    pending_ = false;
    next_update_ = {};
}
void WorkerHeartbeatSignal::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    changed_.notify_all();
}
void WorkerHeartbeatSignal::notify() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stopped_) { pending_ = true; changed_.notify_all(); }
}
void WorkerHeartbeatSignal::begin_update() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Clear before gathering state, so a release during that gather/post stays
    // pending and publishes a subsequent fresh snapshot.
    pending_ = false;
    next_update_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
}
void WorkerHeartbeatSignal::wait(double seconds, bool accept_updates) {
    if (seconds <= 0) return;
    using Clock = std::chrono::steady_clock;
    std::unique_lock<std::mutex> lock(mutex_);
    const auto now = Clock::now();
    const auto maximum = Clock::time_point::max();
    const auto remaining = std::chrono::duration<double>(maximum - now).count();
    const auto deadline = !std::isfinite(seconds) || seconds >= remaining ? maximum
        : now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    while (!stopped_) {
        const auto current = Clock::now();
        if (current >= deadline) return;
        if (accept_updates && pending_) {
            if (current >= next_update_) return;
            changed_.wait_until(lock, std::min(deadline, next_update_), [&] { return stopped_; });
        } else {
            changed_.wait_until(lock, deadline, [&] { return stopped_ || (accept_updates && pending_); });
        }
    }
}
void run_worker_heartbeat(double interval, const WorkerHeartbeatOperations &ops) {
    interval = std::max(2.0, interval);
    bool registered = false;
    double backoff = 2.0;
    try {
    while (!ops.stopped()) {
        if (ops.post("register")) { registered = true; break; }
        ops.wait(backoff);
        backoff = std::min(60.0, backoff * 1.5);
    }
    while (!ops.stopped()) {
        try {
            if (!ops.post("heartbeat")) registered = false;
            if (!registered) registered = ops.post("register");
        } catch (const WorkerHeartbeatCancelled &) {
            throw;
        } catch (...) {
            // Source steady-state state collection errors do not kill the loop.
        }
        if (registered && ops.wait_for_update) ops.wait_for_update(interval);
        else ops.wait(interval);
    }
    } catch (const WorkerHeartbeatCancelled &) {
        // Cancellation does not turn a previously successful registration into
        // a failure. Python still deregisters that worker after cancelling.
    }
    if (registered) {
        try { ops.post("deregister"); } catch (...) {}
    }
}
}
