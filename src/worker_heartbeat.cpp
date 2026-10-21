#include "ezlocalai/worker_heartbeat.hpp"
#include <algorithm>

namespace ezlocalai {
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
        ops.wait(interval);
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
