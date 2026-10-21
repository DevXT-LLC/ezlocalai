#include "ezlocalai/voice_slots.hpp"
#include "whisper_transcription_test.hpp"
#include <future>
#include <thread>
using namespace transcription_test;
thread_local std::string actor = "main";
Json lease_json(const VoiceSlotHandle &lease) {
    const auto s = lease->snapshot(); Json out{{"handoff",s.handoff},{"handoff_state",Json::parse(s.handoff_state)},{"shared_lock",s.shared_lock},{"semaphore",s.semaphore}}; if (s.released) out["released"] = *s.released; return out;
}
Json ordered_events(const Json &events,bool handoff) {
    Json out{{"actors",Json::object()},{"acquired",Json::array()},{"handoff",Json::array()}};
    for (const auto &event : events) {
        const std::string who = event.at("actor"),name = event.at("event");
        if (!out["actors"].contains(who)) out["actors"][who] = Json::array(); out["actors"][who].push_back(event);
        if (name == "acquired") out["acquired"].push_back(event);
        if (handoff && name != "handoff" && name != "acquired" && name != "failed") out["handoff"].push_back(event);
    }
    return out;
}
struct Environment {
    Json config,events = Json::array(),results = Json::object(),active = Json::object(); bool unavailable = false;
    std::mutex mutex;
    std::shared_ptr<VoicePermitQueue> shared = std::make_shared<VoicePermitQueue>(1,true);
    std::map<std::string,std::shared_ptr<VoicePermitQueue>> permits;
    std::map<std::string,std::unique_ptr<VoiceSlotGuard>> guards;
    std::map<std::string,VoiceSlotHandle> leases;
    struct Job { VoiceCancellation cancellation; std::future<void> future; };
    std::map<std::string,Job> jobs;
    VoicePermitQueue idle{0};
    explicit Environment(Json value) : config(std::move(value)) {
        for (const std::string service : {"stt","tts"}) {
            permits[service] = std::make_shared<VoicePermitQueue>(config.at("capacity").get<int64_t>());
            guards[service] = std::make_unique<VoiceSlotGuard>(service,config.at("capacity"),shared,operations(service));
        }
    }
    void event_locked(const std::string &name,Json fields = Json::object()) { fields["event"] = name; fields["actor"] = actor; events.push_back(fields); }
    void event(const std::string &name,Json fields = Json::object()) { std::lock_guard<std::mutex> lock(mutex); event_locked(name,fields); }
    VoiceSlotOperations operations(const std::string &service) {
        VoiceSlotOperations ops;
        ops.should_handoff = [&](const auto &service) { event("handoff",{{"service",service}}); return config.at("handoff").get<bool>(); };
        ops.acquire_permit = [queue = permits.at(service)](const auto &cancel) { queue->acquire(cancel); }; ops.release_permit = [queue = permits.at(service)] { queue->release(); };
        ops.mark_handoff = [&](const auto &service,bool value) { std::lock_guard<std::mutex> lock(mutex); event_locked("mark",{{"service",service},{"value",value}}); active[service] = std::max(0,active.value(service,0) + (value ? 1 : -1)); };
        ops.set_unavailable = [&](bool value) { std::lock_guard<std::mutex> lock(mutex); event_locked("unavailable",{{"value",value}}); unavailable = value; };
        ops.wait_for_idle = [&](const auto &service,const auto &cancel) { event("wait",{{"service",service}}); for (const auto &blocked : config.value("idle_blocked",Json::array())) if (blocked == actor) idle.acquire(cancel); };
        ops.unload = [&](const auto &service,bool force) { event("unload",{{"service",service},{"force",force}}); return config.value("state",Json{{"loaded_models",Json::array({"text"})}}).dump(); };
        ops.destroy = [&](const auto &service,bool async,bool force) { event("destroy",{{"service",service},{"async_cleanup",async},{"force",force}}); };
        ops.restore = [&](const auto &service,const auto &state,const auto &reload) { event("restore",{{"service",service},{"state",Json::parse(state)},{"reload_env",reload}}); };
        return ops;
    }
    void spawn(const std::string &name,const std::string &service) {
        auto &job = jobs[name]; job.future = std::async(std::launch::async,[&,name,service,cancel = job.cancellation] {
            actor = name;
            try {
                auto lease = guards.at(service)->acquire(cancel); std::lock_guard<std::mutex> lock(mutex); leases[name] = lease; results[name] = {{"service",service}}; event_locked("acquired",{{"id",name}});
            } catch (const VoiceCancelled &error) { std::lock_guard<std::mutex> lock(mutex); results[name] = {{"error",{{"category","CancelledError"},{"message",error.what()}}}}; event_locked("failed",{{"id",name},{"category","CancelledError"}}); }
            catch (const std::exception &error) { std::lock_guard<std::mutex> lock(mutex); results[name] = {{"error",{{"category","RuntimeError"},{"message",error.what()}}}}; event_locked("failed",{{"id",name},{"category","RuntimeError"}}); }
        });
    }
    void release(const std::string &name) {
        VoiceSlotHandle lease; std::string service;
        { std::lock_guard<std::mutex> lock(mutex); lease = leases.at(name); service = results.at(name).at("service"); }
        actor = "release-" + name; guards.at(service)->release(lease); actor = "main";
    }
    Json snapshot() {
        const auto shared_state = shared->snapshot(); Json queues = Json::object();
        for (const auto &queue : permits) { const auto s = queue.second->snapshot(); queues[queue.first] = {{"available",s.available},{"waiting",s.waiters.size()}}; }
        std::lock_guard<std::mutex> lock(mutex); auto finished = results; for (const auto &lease : leases) finished[lease.first]["state"] = lease_json(lease.second);
        return {{"state",{{"shared",{{"available",shared_state.available},{"waiting",shared_state.waiters.size()}}},{"permits",queues},{"active",active},{"unavailable",unavailable},{"results",finished}}},{"events",ordered_events(events,config.at("handoff"))}};
    }
    void join() { for (auto &job : jobs) job.second.future.get(); }
};
int main(int argc,char **argv) { try {
    expect(argc == 3,"reference and async voice-slot fixture paths"); const auto cases = records(argv[2]);
    expect(hash(read(fs::path(argv[1])/"Pipes.py")) == cases[0].at("sha256"),"unchanged source voice guard"); expect(cases.size() == cases[0].at("cases").get<size_t>() + 1,"complete real-async fixture"); size_t actions = 0;
    for (size_t row = 1; row < cases.size(); ++row) {
        const auto &scenario = cases[row]; Environment env(scenario.at("config")); size_t step = 0;
        for (const auto &action : scenario.at("actions")) {
            const std::string op = action.at("op"), name = action.value("id",std::string()),service = action.value("service",std::string("stt"));
            if (op == "spawn") env.spawn(name,service); else if (op == "cancel") env.jobs.at(name).cancellation.cancel(); else if (op == "release") env.release(name);
            else if (op == "hold") env.permits.at(service)->acquire(); else if (op == "unhold") env.permits.at(service)->release();
            auto expected = scenario.at("outcomes")[step]; expected["events"] = ordered_events(expected.at("events"),env.config.at("handoff"));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10); Json actual;
            while ((actual = env.snapshot()) != expected && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            // On failure exit immediately: blocked test workers must not hide the diagnostic in a future destructor.
            try { compare_json(actual,expected,"real async voice guard " + std::to_string(row) + " action " + std::to_string(step)); }
            catch (const std::exception &error) { std::cerr << error.what() << '\n' << std::flush; std::_Exit(1); }
            ++step; ++actions;
        }
        env.join();
    }
    std::cout << cases.size() - 1 << " real asyncio scenarios and " << actions << " settled states/FIFO/handoff traces match native threads\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
