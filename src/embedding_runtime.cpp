#include "ezlocalai/embedding_runtime.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ezlocalai_c.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

namespace ezlocalai {
namespace {
std::string lower(std::string value) { for (auto &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return value; }
bool enabled(const char *name) {
    std::string value = ezl_getenv_default(name);
    return lower(strip_config_value(value.empty() ? "true" : value)) == "true";
}
}
bool embedding_keep_loaded() { return enabled("EMBEDDING_KEEP_LOADED"); }
bool embedding_should_preload() {
    std::string remote = ezl_getenv_default("EMBEDDING_SERVER");
    if (lower(remote) == "true") remote.clear();
    while (!remote.empty() && remote.back() == '/') remote.pop_back();
    return enabled("EMBEDDING_ENABLED") && remote.empty() && embedding_keep_loaded();
}
int embedding_parallel_slots() {
    std::string value = ezl_getenv_fallback("EMBEDDING_N_PARALLEL", "1");
    for (const unsigned char c : value) if (c >= 0x1c && c <= 0x1f) return 1;
    value = normalize_decimal_digits(strip_config_value(value));
    if (value.empty()) return 1;
    size_t pos = 0, digits = 0; bool negative = value.front() == '-', previous_digit = false;
    if (value.front() == '-' || value.front() == '+') ++pos;
    int number = 0;
    for (; pos < value.size(); ++pos) {
        const auto c = value[pos];
        if (c == '_' && previous_digit) { previous_digit = false; continue; }
        if (c < '0' || c > '9') return 1;
        const int digit = c - '0';
        number = number > (std::numeric_limits<int>::max() - digit) / 10 ? std::numeric_limits<int>::max() : number * 10 + digit;
        previous_digit = true; ++digits;
    }
    return !previous_digit || digits > 4300 || negative ? 1 : std::max(1, number);
}

EmbeddingPool::EmbeddingPool(size_t capacity, Load load, Destroy destroy, std::function<bool()> keep_loaded, Changed changed)
    : capacity_(std::max<size_t>(1, capacity)), load_(std::move(load)), destroy_(std::move(destroy)), keep_loaded_(std::move(keep_loaded)), residency_changed_(std::move(changed)) {}
void EmbeddingPool::notify_residency_locked() {
    // Observers must not reenter the pool. Publish removals before destruction,
    // and keep concurrent publications in the same order as pool mutations.
    if (residency_changed_) residency_changed_({capacity_, acquired_, waiters_.size(), resident_, transient_, {available_.begin(), available_.end()}});
}
EmbeddingPool::Handle EmbeddingPool::acquire() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto ticket = next_ticket_++;
        waiters_.push_back(ticket);
        changed_.wait(lock, [&] { return waiters_.front() == ticket && acquired_ < capacity_; });
        waiters_.pop_front(); ++acquired_; changed_.notify_all();
    }
    try {
        if (!keep_loaded_()) {
            size_t slot;
            { std::lock_guard<std::mutex> lock(mutex_); slot = resident_.size() + transient_.size() + 1; }
            const auto handle = load_(slot);
            { std::lock_guard<std::mutex> lock(mutex_); transient_.push_back(handle); notify_residency_locked(); }
            return handle;
        }
        auto take_available = [&]() -> std::pair<bool, Handle> {
            std::lock_guard<std::mutex> lock(mutex_);
            if (available_.empty()) return {false, 0};
            const auto handle = available_.front(); available_.pop_front(); return {true, handle};
        };
        auto available = take_available();
        if (available.first) return available.second;
        {
            std::lock_guard<std::mutex> load_lock(load_mutex_);
            available = take_available();
            if (available.first) return available.second;
            size_t count;
            { std::lock_guard<std::mutex> lock(mutex_); count = resident_.size(); }
            if (count < capacity_) {
                const auto handle = load_(count + 1);
                { std::lock_guard<std::mutex> lock(mutex_); resident_.push_back(handle); notify_residency_locked(); }
                return handle;
            }
        }
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return !available_.empty(); });
        const auto handle = available_.front(); available_.pop_front(); return handle;
    } catch (...) { return_permit(); throw; }
}
void EmbeddingPool::return_permit() {
    std::lock_guard<std::mutex> lock(mutex_); --acquired_; changed_.notify_all();
}
void EmbeddingPool::destroy(Handle handle) noexcept {
    try { destroy_(handle); }
    catch (const std::exception &error) { std::cerr << "[Embedding] Cleanup failed: " << error.what() << '\n'; }
    catch (...) { std::cerr << "[Embedding] Cleanup failed\n"; }
}
void EmbeddingPool::release(Handle handle) {
    bool retained = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto resident = std::find(resident_.begin(), resident_.end(), handle);
        retained = keep_loaded_() && resident != resident_.end();
        if (retained) available_.push_back(handle);
        else {
            if (resident != resident_.end()) resident_.erase(resident);
            transient_.erase(std::remove(transient_.begin(), transient_.end(), handle), transient_.end());
            notify_residency_locked();
        }
    }
    if (!retained) destroy(handle);
    return_permit();
}
void EmbeddingPool::warm() {
    std::lock_guard<std::mutex> load_lock(load_mutex_);
    while (true) {
        size_t count;
        { std::lock_guard<std::mutex> lock(mutex_); count = resident_.size(); }
        if (count >= capacity_) return;
        try {
            const auto handle = load_(count + 1);
            std::lock_guard<std::mutex> lock(mutex_);
            resident_.push_back(handle); notify_residency_locked(); available_.push_back(handle); changed_.notify_all();
        } catch (const std::exception &error) {
            std::cerr << "[Embedding] Failed to warm-load instance " << count + 1 << ": " << error.what() << '\n'; return;
        }
    }
}
bool EmbeddingPool::unload_idle() {
    std::lock_guard<std::mutex> load_lock(load_mutex_);
    std::vector<Handle> handles;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (acquired_ || (resident_.empty() && transient_.empty())) return false;
        handles = std::move(resident_); resident_.clear();
        handles.insert(handles.end(), transient_.begin(), transient_.end()); transient_.clear(); available_.clear();
        notify_residency_locked();
    }
    for (const auto handle : handles) destroy(handle);
    return true;
}
EmbeddingPool::Snapshot EmbeddingPool::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {capacity_, acquired_, waiters_.size(), resident_, transient_, {available_.begin(), available_.end()}};
}
}
