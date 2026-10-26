#include "ezlocalai/ezlocalai_c.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
struct Request {
    std::string id;
    std::string completion_type;
    std::string payload;
    std::string result;
    std::string error;
    double timestamp = 0;
    unsigned long long sequence = 0;
    ezl_request_status status = EZL_REQUEST_QUEUED;
    bool done = false;
    bool cancelled = false;
};
}

struct ezl_queue {
    size_t max_concurrent_requests = 1;
    size_t max_queue_size = 1;
    size_t processing = 0;
    size_t active = 0;
    size_t terminal_count = 0;
    unsigned long long total_queued = 0;
    unsigned long long total_processed = 0;
    unsigned long long total_failed = 0;
    unsigned long long next_id = 1;
    bool started = false;
    bool stopping = false;
    ezl_processor_fn processor = nullptr;
    void *user_data = nullptr;
    ezl_queue_state_fn state_observer = nullptr;
    void *state_observer_data = nullptr;
    std::mutex observer_mutex;
    std::vector<std::thread> workers;
    // Serialize lifecycle operations separately so stop can join workers without
    // holding the mutex they need to finish requests.
    std::mutex lifecycle_mutex;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::shared_ptr<Request>> pending;
    std::unordered_map<std::string, std::shared_ptr<Request>> requests;
};

namespace {
void notify_state(ezl_queue *queue) noexcept {
    // Separate from the state mutex: observers may read queue snapshots.
    // The setter synchronizes with this lock before replacing user_data.
    std::lock_guard<std::mutex> lock(queue->observer_mutex);
    // An observational callback cannot turn completed inference into a queue
    // failure. NativeServer only schedules a fresh heartbeat here.
    if (queue->state_observer) { try { queue->state_observer(queue->state_observer_data); } catch (...) {} }
}
// Bound all terminal outcomes. Python's history target is 1000, but its
// success-only pruning leaves failure-heavy workloads unbounded.
constexpr size_t history_limit = 1000;

void prune_history(ezl_queue *queue) {
    while (queue->terminal_count > history_limit) {
        auto oldest = queue->requests.end();
        for (auto it = queue->requests.begin(); it != queue->requests.end(); ++it) {
            if (!it->second->done) continue;
            if (oldest == queue->requests.end()
                || it->second->timestamp < oldest->second->timestamp
                || (it->second->timestamp == oldest->second->timestamp
                    && it->second->sequence < oldest->second->sequence)) oldest = it;
        }
        if (oldest == queue->requests.end()) break;
        // Existing waiters own a shared_ptr and can still collect the result.
        queue->requests.erase(oldest);
        --queue->terminal_count;
    }
}

size_t result_buffer_size() {
    const char *raw = std::getenv("NATIVE_QUEUE_RESULT_BYTES");
    if (!raw || !*raw) return 4u * 1024u * 1024u;
    char *end = nullptr;
    const auto parsed = std::strtoul(raw, &end, 10);
    if (end == raw || parsed < 65536ul) return 4u * 1024u * 1024u;
    return std::min(parsed, 128ul * 1024ul * 1024ul);
}

std::shared_ptr<Request> find_request(ezl_queue *queue, const char *id) {
    const auto it = queue->requests.find(id);
    return it == queue->requests.end() ? nullptr : it->second;
}

void worker_main(ezl_queue *queue) {
    for (;;) {
        std::unique_lock<std::mutex> lock(queue->mutex);
        queue->changed.wait(lock, [&] {
            return (queue->stopping && queue->pending.empty())
                || (!queue->pending.empty() && queue->processing < queue->max_concurrent_requests);
        });
        if (queue->pending.empty()) return;
        auto request = queue->pending.front();
        queue->pending.pop_front();
        ++queue->processing;
        const bool cancelled = request->cancelled;
        request->status = cancelled ? EZL_REQUEST_CANCELLED : EZL_REQUEST_PROCESSING;
        queue->changed.notify_all();
        lock.unlock();
        notify_state(queue);

        std::string result;
        std::string error;
        bool failed = false;
        if (!cancelled) {
            try {
                std::vector<char> buffer(result_buffer_size(), '\0');
                if (queue->processor) queue->processor(request->payload.c_str(), buffer.data(), buffer.size(), queue->user_data);
                else std::snprintf(buffer.data(), buffer.size(), "%s", request->payload.c_str());
                buffer.back() = '\0';
                result = buffer.data();
            } catch (const std::bad_alloc &) {
                failed = true;
                error = "out of memory";
            } catch (const std::exception &exception) {
                failed = true;
                error = exception.what();
            } catch (...) {
                failed = true;
                error = "request processor failed";
            }
        }

        lock.lock();
        if (!cancelled) {
            if (!failed) {
                request->result = std::move(result);
                request->status = EZL_REQUEST_COMPLETED;
                ++queue->total_processed;
            } else {
                request->error = std::move(error);
                request->status = EZL_REQUEST_FAILED;
                ++queue->total_failed;
            }
        }
        request->done = true;
        // Input may contain large images/audio. Status and result replay do not
        // need it once the callback has returned; clear() alone retains capacity.
        std::string{}.swap(request->payload);
        ++queue->terminal_count;
        prune_history(queue);
        --queue->processing;
        --queue->active;
        queue->changed.notify_all();
        lock.unlock();
        notify_state(queue);
    }
}
}

ezl_queue *ezl_queue_create(size_t concurrency, size_t capacity) {
    try {
        auto queue = std::make_unique<ezl_queue>();
        queue->max_concurrent_requests = std::max(size_t{1}, concurrency);
        queue->max_queue_size = std::max(size_t{1}, capacity);
        return queue.release();
    } catch (...) { return nullptr; }
}

void ezl_queue_destroy(ezl_queue *queue) {
    if (!queue) return;
    ezl_queue_stop(queue);
    delete queue;
}

int ezl_queue_start(ezl_queue *queue, ezl_processor_fn processor, void *user_data) {
    if (!queue) return EINVAL;
    std::lock_guard<std::mutex> lifecycle(queue->lifecycle_mutex);
    std::unique_lock<std::mutex> lock(queue->mutex);
    if (queue->started) return 0;
    queue->processor = processor;
    queue->user_data = user_data;
    queue->stopping = false;
    int error = 0;
    try {
        queue->workers.reserve(queue->max_concurrent_requests);
        for (size_t i = 0; i < queue->max_concurrent_requests; ++i) queue->workers.emplace_back(worker_main, queue);
    } catch (const std::system_error &exception) { error = exception.code().value(); }
    catch (const std::bad_alloc &) { error = ENOMEM; }
    catch (const std::length_error &) { error = ENOMEM; }
    if (!error) { queue->started = true; return 0; }
    queue->stopping = true;
    queue->changed.notify_all();
    lock.unlock();
    for (auto &worker : queue->workers) worker.join();
    queue->workers.clear();
    return error;
}

void ezl_queue_stop(ezl_queue *queue) {
    if (!queue) return;
    std::lock_guard<std::mutex> lifecycle(queue->lifecycle_mutex);
    {
        std::lock_guard<std::mutex> lock(queue->mutex);
        if (!queue->started) return;
        queue->stopping = true;
        queue->changed.notify_all();
    }
    for (auto &worker : queue->workers) worker.join();
    queue->workers.clear();
    std::lock_guard<std::mutex> lock(queue->mutex);
    queue->started = false;
}

int ezl_queue_is_running(ezl_queue *queue) {
    if (!queue) return 0;
    std::lock_guard<std::mutex> lock(queue->mutex);
    return queue->started && !queue->stopping;
}

int ezl_queue_set_concurrency(ezl_queue *queue, size_t concurrency) {
    if (!queue) return EINVAL;
    std::unique_lock<std::mutex> lock(queue->mutex);
    if (queue->started && queue->stopping) return ECANCELED;
    concurrency = std::max<size_t>(1, concurrency);
    // Workers above a reduced limit remain idle and can serve a later growth.
    // Stop prevents further growth before joining them outside this mutex.
    try {
        if (queue->started) {
            queue->workers.reserve(concurrency);
            while (queue->workers.size() < concurrency) queue->workers.emplace_back(worker_main, queue);
        }
    } catch (const std::system_error &error) { return error.code().value(); }
    catch (const std::bad_alloc &) { return ENOMEM; }
    catch (const std::length_error &) { return ENOMEM; }
    const bool changed = queue->max_concurrent_requests != concurrency;
    queue->max_concurrent_requests = concurrency;
    queue->changed.notify_all();
    lock.unlock();
    if (changed) notify_state(queue);
    return 0;
}

int ezl_queue_enqueue(ezl_queue *queue, const char *payload, char *out, size_t size) {
    return ezl_queue_enqueue_typed(queue, payload, "request", out, size);
}

int ezl_queue_enqueue_typed(ezl_queue *queue, const char *payload, const char *type, char *out, size_t size) {
    if (!queue || !payload || !out || !size) return EINVAL;
    try {
        auto request = std::make_shared<Request>();
        request->payload = payload;
        request->completion_type = type ? type : "request";
        request->timestamp = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::unique_lock<std::mutex> lock(queue->mutex);
        if (!queue->started || queue->stopping) return EINVAL;
        if (queue->pending.size() >= queue->max_queue_size) return ENOSPC;
        request->sequence = queue->next_id++;
        request->id = "ezl-" + std::to_string(request->sequence);
        queue->requests.emplace(request->id, request);
        try { queue->pending.push_back(request); }
        catch (...) { queue->requests.erase(request->id); throw; }
        std::snprintf(out, size, "%s", request->id.c_str());
        ++queue->active;
        ++queue->total_queued;
        queue->changed.notify_all();
        lock.unlock();
        notify_state(queue);
        return 0;
    } catch (const std::bad_alloc &) { return ENOMEM; }
}

int ezl_queue_wait(ezl_queue *queue, const char *id, long timeout_ms, char *result, size_t result_size, char *error, size_t error_size) {
    if (!queue || !id) return EINVAL;
    if (result && result_size) result[0] = '\0';
    if (error && error_size) error[0] = '\0';
    std::unique_lock<std::mutex> lock(queue->mutex);
    const auto request = find_request(queue, id);
    if (!request) return ENOENT;
    if (timeout_ms > 0) {
        if (!queue->changed.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return request->done; })) return ETIMEDOUT;
    } else queue->changed.wait(lock, [&] { return request->done; });
    if (request->status == EZL_REQUEST_FAILED) {
        if (error && error_size) std::snprintf(error, error_size, "%s", request->error.c_str());
        return EIO;
    }
    if (request->status == EZL_REQUEST_CANCELLED) {
        if (error && error_size) std::snprintf(error, error_size, "request cancelled");
        return ECANCELED;
    }
    if (result && result_size) std::snprintf(result, result_size, "%s", request->result.c_str());
    return 0;
}

int ezl_queue_cancel(ezl_queue *queue, const char *id) {
    if (!queue || !id) return EINVAL;
    std::unique_lock<std::mutex> lock(queue->mutex);
    const auto request = find_request(queue, id);
    if (!request) return ENOENT;
    request->cancelled = true;
    if (request->status == EZL_REQUEST_QUEUED) request->status = EZL_REQUEST_CANCELLED;
    queue->changed.notify_all();
    lock.unlock();
    notify_state(queue);
    return 0;
}

int ezl_queue_set_state_observer(ezl_queue *queue, ezl_queue_state_fn observer, void *user_data) {
    if (!queue) return EINVAL;
    std::lock_guard<std::mutex> lifecycle(queue->lifecycle_mutex);
    {
        std::lock_guard<std::mutex> lock(queue->mutex);
        if (queue->started) return EBUSY;
    }
    // Also waits for an observer already in progress, so stopped callers may
    // release the previous user_data once unregistration returns.
    std::lock_guard<std::mutex> observer_lock(queue->observer_mutex);
    queue->state_observer = observer;
    queue->state_observer_data = user_data;
    return 0;
}

int ezl_queue_get_request_status(ezl_queue *queue, const char *id, ezl_request_status *out) {
    if (!queue || !id || !out) return EINVAL;
    std::lock_guard<std::mutex> lock(queue->mutex);
    const auto request = find_request(queue, id);
    if (!request) return ENOENT;
    *out = request->status;
    return 0;
}

int ezl_queue_get_request_info(ezl_queue *queue, const char *id, ezl_request_info *out) {
    if (!queue || !id || !out) return EINVAL;
    std::lock_guard<std::mutex> lock(queue->mutex);
    const auto request = find_request(queue, id);
    if (!request) return ENOENT;
    *out = {};
    std::snprintf(out->id, sizeof(out->id), "%s", request->id.c_str());
    std::snprintf(out->completion_type, sizeof(out->completion_type), "%s", request->completion_type.c_str());
    std::snprintf(out->error, sizeof(out->error), "%s", request->error.c_str());
    out->status = request->status;
    out->timestamp = request->timestamp;
    return 0;
}

void ezl_queue_get_snapshot(ezl_queue *queue, ezl_queue_snapshot *out) {
    if (!queue || !out) return;
    std::lock_guard<std::mutex> lock(queue->mutex);
    *out = {queue->max_queue_size, queue->max_concurrent_requests, queue->pending.size(), queue->processing, queue->active,
            queue->total_queued, queue->total_processed, queue->total_failed};
}

const char *ezl_request_status_name(ezl_request_status status) {
    switch (status) {
        case EZL_REQUEST_QUEUED: return "queued";
        case EZL_REQUEST_PROCESSING: return "processing";
        case EZL_REQUEST_COMPLETED: return "completed";
        case EZL_REQUEST_FAILED: return "failed";
        case EZL_REQUEST_CANCELLED: return "cancelled";
        default: return "unknown";
    }
}
