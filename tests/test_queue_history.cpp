#include "ezlocalai/ezlocalai_c.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false;
    std::atomic<int> calls{0};
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "processor did not start");
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }
};
void processor(const char *input, char *output, size_t size, void *opaque) {
    auto &gate = *static_cast<Gate *>(opaque);
    ++gate.calls;
    if (std::strcmp(input, "block") == 0) {
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.entered = true;
        gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(20), [&] { return gate.released; }))
            throw std::runtime_error("gate timed out");
    }
    if (std::strcmp(input, "fail") == 0) throw std::runtime_error("expected failure");
    std::snprintf(output, size, "result");
}
using Queue = std::unique_ptr<ezl_queue, decltype(&ezl_queue_destroy)>;
Queue start(Gate &gate, size_t concurrency = 1) {
    Queue queue(ezl_queue_create(concurrency, 1200), ezl_queue_destroy);
    require(queue && ezl_queue_start(queue.get(), processor, &gate) == 0, "queue start failed");
    return queue;
}
std::string enqueue(ezl_queue *queue, const char *input) {
    char id[128];
    require(ezl_queue_enqueue_typed(queue, input, "chat", id, sizeof(id)) == 0, "enqueue failed");
    return id;
}
int wait(ezl_queue *queue, const std::string &id) {
    char result[128], error[128];
    const int code = ezl_queue_wait(queue, id.c_str(), 10000, result, sizeof(result), error, sizeof(error));
    if (!code) require(std::strcmp(result, "result") == 0, "result corrupted");
    if (code == EIO) require(std::strcmp(error, "expected failure") == 0, "error corrupted");
    return code;
}
void check_history(ezl_queue *queue, const std::vector<std::string> &ids, ezl_request_status expected) {
    size_t retained = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        ezl_request_info info{};
        const int code = ezl_queue_get_request_info(queue, ids[i].c_str(), &info);
        if (i + 1000 < ids.size()) require(code == ENOENT, "old terminal request retained");
        else {
            require(code == 0 && info.status == expected, "recent history lost or changed");
            require(std::strcmp(info.completion_type, "chat") == 0, "request metadata lost");
            ++retained;
        }
    }
    require(retained == 1000, "incorrect history limit");
}
void serial(bool failure) {
    Gate gate;
    auto queue = start(gate);
    std::vector<std::string> ids;
    // Large inputs with tiny results exercise payload disposal independently of
    // eviction: retained history must not own the original inference inputs.
    const std::string input = failure ? "fail" : std::string(256 * 1024, 'x');
    for (int i = 0; i < 1105; ++i) {
        ids.push_back(enqueue(queue.get(), input.c_str()));
        require(wait(queue.get(), ids.back()) == (failure ? EIO : 0), "serial completion failed");
    }
    check_history(queue.get(), ids, failure ? EZL_REQUEST_FAILED : EZL_REQUEST_COMPLETED);
    require(wait(queue.get(), ids.back()) == (failure ? EIO : 0), "recent result replay failed");
    ezl_queue_snapshot snapshot{};
    ezl_queue_get_snapshot(queue.get(), &snapshot);
    require(snapshot.active == 0 && snapshot.total_queued == 1105, "queue accounting changed");
    require(snapshot.total_processed == (failure ? 0 : 1105) && snapshot.total_failed == (failure ? 1105 : 0), "terminal counters changed");
    ezl_queue_stop(queue.get());
    require(ezl_queue_start(queue.get(), processor, &gate) == 0, "restart failed");
    ids.push_back(enqueue(queue.get(), input.c_str()));
    require(wait(queue.get(), ids.back()) == (failure ? EIO : 0), "restart completion failed");
    check_history(queue.get(), ids, failure ? EZL_REQUEST_FAILED : EZL_REQUEST_COMPLETED);
}
void cancelled() {
    Gate gate;
    auto queue = start(gate);
    const auto blocked = enqueue(queue.get(), "block");
    gate.wait_entered();
    std::vector<std::string> ids;
    for (int i = 0; i < 1105; ++i) {
        ids.push_back(enqueue(queue.get(), "unused"));
        require(ezl_queue_cancel(queue.get(), ids.back().c_str()) == 0, "cancel failed");
    }
    gate.release();
    ezl_queue_stop(queue.get());
    require(gate.calls == 1, "cancelled work executed");
    check_history(queue.get(), ids, EZL_REQUEST_CANCELLED);
    require(wait(queue.get(), ids.back()) == ECANCELED, "cancelled result changed");
    ezl_queue_snapshot snapshot{};
    ezl_queue_get_snapshot(queue.get(), &snapshot);
    require(snapshot.active == 0 && snapshot.total_processed == 1 && snapshot.total_failed == 0, "cancel accounting changed");
}
void active_waiter() {
    Gate gate;
    auto queue = start(gate, 2);
    const auto blocked = enqueue(queue.get(), "block");
    gate.wait_entered();
    std::promise<void> ready;
    auto registered = ready.get_future();
    auto waiter = std::async(std::launch::async, [&] {
        ready.set_value();
        return wait(queue.get(), blocked);
    });
    registered.wait();
    std::vector<std::string> ids;
    for (int i = 0; i < 1105; ++i) {
        ids.push_back(enqueue(queue.get(), "ok"));
        require(wait(queue.get(), ids.back()) == 0, "concurrent completion failed");
    }
    ezl_request_status status{};
    require(ezl_queue_get_request_status(queue.get(), blocked.c_str(), &status) == 0 && status == EZL_REQUEST_PROCESSING, "active request evicted");
    require(waiter.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout, "blocked waiter completed early");
    gate.release();
    require(waiter.get() == 0, "registered waiter lost evicted result");
    require(ezl_queue_get_request_status(queue.get(), blocked.c_str(), &status) == ENOENT, "oldest enqueue not evicted after completion");
    check_history(queue.get(), ids, EZL_REQUEST_COMPLETED);
}
}
int main() {
    try {
        serial(false);
        serial(true);
        cancelled();
        active_waiter();
        std::cout << "queue history: success, failure, cancellation, restart and active waiter passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
