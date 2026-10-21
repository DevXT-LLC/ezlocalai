#include "ezlocalai/model_lifecycle.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ezlocalai {
namespace {
double rounded(double value) {
    if (!std::isfinite(value)) return value;
    // Round the binary floating-point value directly to four decimal places.
    // Multiplying by 10000 first changes Python's halfway rounding behavior.
    char buffer[512];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::fixed, 4);
    if (result.ec != std::errc()) throw std::runtime_error("Cannot round lifecycle duration");
    double out;
    const auto parsed = std::from_chars(buffer, result.ptr, out, std::chars_format::fixed);
    if (parsed.ec != std::errc()) throw std::runtime_error("Cannot parse lifecycle duration");
    return out;
}
}
ModelLifecycle::ModelLifecycle(std::function<double()> clock) : clock_(std::move(clock)) {
    if (!clock_) clock_ = [] { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); };
}
void ModelLifecycle::record(const std::string &operation, const std::string &model_type,
                            const std::string &model, double duration, bool success) {
    duration = std::max(0.0, duration);
    ModelLifecycleEvent event{clock_(), rounded(duration), operation, model_type, model.empty() ? "unknown" : model, success};
    const auto key = model_type + ":" + operation + ":" + model;
    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back(std::move(event));
    if (events_.size() > 100) events_.pop_front();
    auto found = indices_.find(key);
    if (found == indices_.end()) {
        found = indices_.emplace(key, aggregates_.size()).first;
        aggregates_.push_back({{operation, model_type, model.empty() ? "unknown" : model}, 0});
    }
    auto &aggregate = aggregates_[found->second];
    auto &summary = aggregate.summary;
    ++summary.count;
    if (!success) ++summary.failures;
    aggregate.total += duration;
    summary.min_seconds = summary.count == 1 ? duration : std::min(summary.min_seconds, duration);
    summary.max_seconds = std::max(summary.max_seconds, duration);
    summary.last_seconds = duration;
}
ModelLifecycleSnapshot ModelLifecycle::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ModelLifecycleSnapshot result;
    result.recent_events.assign(events_.begin(), events_.end());
    for (const auto &aggregate : aggregates_) {
        auto summary = aggregate.summary;
        summary.average_seconds = rounded(aggregate.total / std::max<uint64_t>(1, summary.count));
        summary.min_seconds = rounded(summary.min_seconds);
        summary.max_seconds = rounded(summary.max_seconds);
        summary.last_seconds = rounded(summary.last_seconds);
        result.summary.push_back(std::move(summary));
    }
    return result;
}
}
