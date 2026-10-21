#pragma once
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ezlocalai {
struct ModelLifecycleEvent {
    double timestamp, duration_seconds;
    std::string operation, model_type, model;
    bool success;
};
struct ModelLifecycleSummary {
    std::string operation, model_type, model;
    uint64_t count = 0, failures = 0;
    double min_seconds = 0, max_seconds = 0, last_seconds = 0, average_seconds = 0;
};
struct ModelLifecycleSnapshot {
    std::vector<ModelLifecycleSummary> summary;
    std::vector<ModelLifecycleEvent> recent_events;
};
class ModelLifecycle {
public:
    explicit ModelLifecycle(std::function<double()> clock = {});
    void record(const std::string &operation, const std::string &model_type,
                const std::string &model, double duration, bool success = true);
    ModelLifecycleSnapshot snapshot() const;
private:
    struct Aggregate { ModelLifecycleSummary summary; double total = 0; };
    std::function<double()> clock_;
    mutable std::mutex mutex_;
    std::deque<ModelLifecycleEvent> events_;
    std::vector<Aggregate> aggregates_;
    std::unordered_map<std::string, size_t> indices_;
};
}
