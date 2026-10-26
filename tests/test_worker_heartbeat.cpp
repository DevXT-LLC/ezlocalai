#include "ezlocalai/worker_heartbeat.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <atomic>
#include <future>
#include <thread>

namespace {
using namespace std::chrono_literals;
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
struct StopSignal { ezlocalai::WorkerHeartbeatSignal &signal; ~StopSignal() { signal.stop(); } };
void signal_contracts() {
    ezlocalai::WorkerHeartbeatSignal signal;
    signal.start(); signal.begin_update();
    const auto start = std::chrono::steady_clock::now();
    auto notified = std::async(std::launch::async, [&] { signal.wait(30, true); });
    StopSignal stop{signal};
    for (int i = 0; i < 1000; ++i) signal.notify();
    check(notified.wait_for(1s) == std::future_status::ready, "state events must wake the periodic wait");
    notified.get();
    check(std::chrono::steady_clock::now() - start >= 20ms, "a notification burst must respect coalescing spacing");

    signal.begin_update();
    auto backoff = std::async(std::launch::async, [&] { signal.wait(.1, false); });
    signal.notify();
    check(backoff.wait_for(20ms) != std::future_status::ready, "state events must not shorten registration backoff");
    check(backoff.wait_for(1s) == std::future_status::ready, "bounded backoff must expire");
    backoff.get();
    auto cancelled = std::async(std::launch::async, [&] { signal.wait(30, false); });
    signal.stop();
    check(cancelled.wait_for(1s) == std::future_status::ready, "shutdown must interrupt even a backoff wait");
    cancelled.get();
}
void update_during_post() {
    ezlocalai::WorkerHeartbeatSignal signal; signal.start();
    std::atomic<bool> stopped{false}, busy{true};
    std::mutex mutex; std::condition_variable changed; bool captured = false, release = false;
    std::vector<bool> observed;
    ezlocalai::WorkerHeartbeatOperations ops;
    ops.stopped = [&] { return stopped.load(); };
    ops.wait = [&](double seconds) { signal.wait(seconds, false); };
    ops.wait_for_update = [&](double seconds) { signal.wait(seconds, true); };
    ops.post = [&](const std::string &action) {
        if (action == "deregister") return true;
        signal.begin_update();
        if (action == "heartbeat") {
            std::unique_lock<std::mutex> lock(mutex);
            observed.push_back(busy.load());
            if (observed.size() == 1) {
                captured = true; changed.notify_all(); changed.wait(lock, [&] { return release; });
            } else { stopped = true; signal.stop(); }
        }
        return true;
    };
    auto runner = std::async(std::launch::async, [&] { ezlocalai::run_worker_heartbeat(30, ops); });
    struct Cleanup {
        ezlocalai::WorkerHeartbeatSignal &signal; std::atomic<bool> &stopped;
        std::mutex &mutex; std::condition_variable &changed; bool &release;
        ~Cleanup() { stopped = true; signal.stop(); std::lock_guard<std::mutex> lock(mutex); release = true; changed.notify_all(); }
    } cleanup{signal, stopped, mutex, changed, release};
    {
        std::unique_lock<std::mutex> lock(mutex);
        check(changed.wait_for(lock, 1s, [&] { return captured; }), "first heartbeat captures busy state");
        busy = false; signal.notify(); release = true; changed.notify_all();
    }
    check(runner.wait_for(1s) == std::future_status::ready, "a release during HTTP posting must produce another fresh update");
    runner.get();
    check(observed == std::vector<bool>({true, false}), "completion state must not be lost behind the posted busy snapshot");
}
void retry_wait_contract() {
    ezlocalai::WorkerHeartbeatOperations ops;
    bool stopped = false; int posts = 0, ordinary_waits = 0, update_waits = 0;
    ops.stopped = [&] { return stopped; };
    // Initial registration failure, registration success, failed heartbeat and
    // failed re-registration, then successful heartbeat plus registration.
    const std::vector<bool> outcomes{false, true, false, false, true, true};
    ops.post = [&](const std::string &action) { return action == "deregister" || outcomes.at(posts++); };
    ops.wait = [&](double) { ++ordinary_waits; };
    ops.wait_for_update = [&](double) { ++update_waits; stopped = true; };
    ezlocalai::run_worker_heartbeat(30, ops);
    check(ordinary_waits == 2 && update_waits == 1, "only registered steady state may consume update events");
}
}

int main(int argc, char **argv) {
    using Json = nlohmann::json;
    try {
        if (argc != 2) throw std::runtime_error("reference fixture required");
        signal_contracts(); update_during_post(); retry_wait_contract();
        std::ifstream file(argv[1]);
        const auto fixture = Json::parse(file);
        for (const auto &test : fixture.at("cases")) {
            Json trace = Json::array();
            size_t post = 0, waits = 0;
            bool stopped = false;
            ezlocalai::WorkerHeartbeatOperations ops;
            ops.stopped = [&] { return stopped; };
            ops.post = [&](const std::string &path) {
                const auto &outcomes = test.at("outcomes");
                const bool ok = post < outcomes.size() ? outcomes.at(post).get<bool>() : true;
                ++post; trace.push_back({{"post", path}, {"ok", ok}}); return ok;
            };
            ops.wait = [&](double seconds) {
                trace.push_back({{"wait", seconds}});
                stopped = ++waits >= test.at("wait_limit").get<size_t>();
            };
            ezlocalai::run_worker_heartbeat(test.at("interval"), ops);
            if (trace != test.at("trace")) throw std::runtime_error(test.at("name").get<std::string>() + ": heartbeat trace differs: " + trace.dump());
        }
        std::cout << fixture.at("cases").size() << " complete Python heartbeat/recovery/shutdown traces and event/coalescing contracts passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
