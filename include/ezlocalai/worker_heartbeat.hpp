#pragma once
#include <functional>
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
};
// Initial registration backs off; steady-state recovery remains on the
// captured heartbeat interval, including after failed re-registration.
void run_worker_heartbeat(double interval, const WorkerHeartbeatOperations &operations);
}
