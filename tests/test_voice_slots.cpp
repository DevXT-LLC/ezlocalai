#include "ezlocalai/voice_slots.hpp"
#include "whisper_transcription_test.hpp"
#include <atomic>
#include <future>
#include <thread>
using namespace transcription_test;
Json state_json(const VoiceSlotHandle &lease) {
    const auto state = lease->snapshot(); Json out{{"handoff",state.handoff},{"handoff_state",Json::parse(state.handoff_state)},{"shared_lock",state.shared_lock},{"semaphore",state.semaphore}};
    if (state.released) out["released"] = *state.released; return out;
}
struct Environment {
    Json config,events = Json::array(); std::map<std::string,int> counts;
    int64_t shared = 1, semaphore; int active = 0; bool unavailable = false,handoff;
    std::map<std::string,VoiceSlotHandle> leases;
    explicit Environment(Json value) : config(std::move(value)),semaphore(std::max<int64_t>(1,config.at("capacity").get<int64_t>())),handoff(config.at("handoff")) {}
    void event(const std::string &name,Json fields = Json::object()) {
        fields["event"] = name; events.push_back(fields); const auto count = ++counts[name];
        for (const auto &failure : config.value("failures",Json::array({config.value("failure",Json::object())})))
            if (failure.value("event",std::string()) == name && failure.value("at",1) == count) {
                if (failure.value("cancel",false)) throw VoiceCancelled(name + " cancelled"); throw std::runtime_error(name + " failure");
            }
    }
    VoiceSlotOperations operations() {
        VoiceSlotOperations ops;
        ops.should_handoff = [&](const auto &service) { event("handoff",{{"service",service}}); return handoff; };
        ops.acquire_shared = [&](const auto &) { event("shared.acquire"); if (!shared) throw std::runtime_error("fixture unexpected blocking acquisition"); --shared; };
        ops.release_shared = [&] { event("shared.release"); ++shared; };
        ops.acquire_permit = [&](const auto &) { event("semaphore.acquire"); if (!semaphore) throw std::runtime_error("fixture unexpected blocking acquisition"); --semaphore; };
        ops.release_permit = [&] { event("semaphore.release"); ++semaphore; };
        ops.mark_handoff = [&](const auto &service,bool value) { event(value ? "mark.true" : "mark.false",{{"service",service}}); active = std::max(0,active + (value ? 1 : -1)); };
        ops.set_unavailable = [&](bool value) { event("unavailable",{{"value",value}}); unavailable = value; };
        ops.wait_for_idle = [&](const auto &service,const auto &) { event("wait",{{"service",service}}); };
        ops.unload = [&](const auto &service,bool force) { event("unload",{{"service",service},{"force",force}}); return config.at("state").dump(); };
        ops.destroy = [&](const auto &service,bool async,bool force) { event("destroy." + service,{{"async_cleanup",async},{"force",force}}); };
        ops.restore = [&](const auto &service,const auto &state,const auto &reload) { event("restore",{{"service",service},{"state",Json::parse(state)},{"reload_env",reload}}); };
        return ops;
    }
    Json snapshot(const VoiceSlotGuard &guard) const {
        Json out{{"shared",shared},{"semaphore",semaphore},{"active",active},{"unavailable",unavailable},{"leases",Json::object()},{"stacks",Json::object()}};
        for (const auto &lease : leases) out["leases"][lease.first] = state_json(lease.second);
        for (const auto &stack : guard.stacks()) { auto values = Json::array(); for (const auto &lease : stack.second) values.push_back(state_json(lease)); out["stacks"][std::to_string(stack.first)] = values; }
        return out;
    }
};
template<class Predicate> void until(Predicate predicate,const std::string &label) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!predicate()) { expect(std::chrono::steady_clock::now() < deadline,label); std::this_thread::yield(); }
}
void concurrency() {
    for (bool binary : {false,true}) for (size_t cancelled : {0,1,2}) {
        VoicePermitQueue queue(1,binary); queue.acquire(); std::vector<int> order; std::mutex mutex;
        VoiceCancellation cancellation[3]; std::vector<std::future<void>> tasks;
        for (int i = 0; i < 3; ++i) {
            tasks.push_back(std::async(std::launch::async,[&,i] { try { queue.acquire(cancellation[i]); { std::lock_guard<std::mutex> lock(mutex); order.push_back(i); } queue.release(); } catch (const VoiceCancelled &) { expect(size_t(i) == cancelled,"only requested waiter cancelled"); } }));
            until([&] { return queue.snapshot().waiters.size() == size_t(i + 1); },"enqueue ordered waiters");
        }
        cancellation[cancelled].cancel(); tasks[cancelled].get(); expect(queue.snapshot().waiters.size() == 2,"cancelled waiter removed"); queue.release();
        for (size_t i = 0; i < tasks.size(); ++i) if (i != cancelled) tasks[i].get();
        std::vector<int> expected; for (int i = 0; i < 3; ++i) if (size_t(i) != cancelled) expected.push_back(i);
        expect(order == expected && queue.snapshot().available == 1 && queue.snapshot().waiters.empty(),"FIFO survives head/middle/tail cancellation");
    }
    auto shared = std::make_shared<VoicePermitQueue>(1,true); std::atomic<int> destroys{0},restores{0}; VoiceSlotOperations ops;
    ops.should_handoff = [](const auto &) { return true; }; ops.destroy = [&](const auto &,bool,bool) { ++destroys; }; ops.restore = [&](const auto &,const auto &,const auto &) { ++restores; };
    VoiceSlotGuard guard("stt",1,shared,ops); const auto lease = guard.acquire(); std::vector<std::thread> releases;
    for (int i = 0; i < 16; ++i) releases.emplace_back([&] { guard.release(lease); }); for (auto &thread : releases) thread.join();
    expect(destroys == 1 && restores == 1 && guard.permits().available == 1 && shared->snapshot().available == 1,"cross-thread duplicate release occurs once");
    // Repeated races cover cancellation before/after wakeup without losing a permit.
    for (int i = 0; i < 100; ++i) {
        VoicePermitQueue queue(1); queue.acquire(); VoiceCancellation cancel;
        auto worker = std::async(std::launch::async,[&] { try { queue.acquire(cancel); queue.release(); } catch (const VoiceCancelled &) {} });
        until([&] { return !queue.snapshot().waiters.empty(); },"queue cancellation race");
        std::thread cancellation([&] { cancel.cancel(); }); queue.release(); cancellation.join(); worker.get();
        expect(queue.snapshot().available == 1 && queue.snapshot().waiters.empty(),"cancellation/wakeup race conserves permits");
    }
}
int main(int argc,char **argv) { try {
    expect(argc == 3,"reference and voice-slot fixture paths"); const auto cases = records(argv[2]);
    expect(hash(read(fs::path(argv[1])/"Pipes.py")) == cases[0].at("sha256"),"unchanged source voice guard"); expect(cases.size() == cases[0].at("cases").get<size_t>() + 1,"complete voice-slot fixture"); size_t actions = 0;
    for (size_t row = 1; row < cases.size(); ++row) {
        const auto &scenario = cases[row]; Environment env(scenario.at("config")); VoiceSlotGuard guard(env.config.at("service"),env.config.at("capacity"),std::make_shared<VoicePermitQueue>(1,true),env.operations());
        uint64_t key = 0; size_t step = 0;
        for (const auto &action : scenario.at("actions")) {
            key = action.value("key",key); Json result = Json::object(); const auto op = action.at("op").get<std::string>();
            try {
                if (op == "acquire") { auto lease = guard.acquire(); env.leases[action.at("id")] = lease; result["value"] = state_json(lease); }
                else if (op == "release") { const auto at = env.leases.find(action.value("id",std::string())); guard.release(at == env.leases.end() ? VoiceSlotHandle{} : at->second); }
                else if (op == "enter") { guard.enter(key); result["value"] = "guard"; }
                else if (op == "exit") result["value"] = guard.exit(key);
                else if (op == "handoff") env.handoff = action.at("value");
            } catch (const VoiceCancelled &error) { result["error"] = {{"category","CancelledError"},{"message",error.what()}}; }
            catch (const std::exception &error) { result["error"] = {{"category","RuntimeError"},{"message",error.what()}}; }
            result["state"] = env.snapshot(guard); compare_json(result,scenario.at("outcomes")[step],"voice slot " + std::to_string(row) + " action " + std::to_string(step)); ++step; ++actions;
        }
        compare_json(env.events,scenario.at("events"),"voice slot " + std::to_string(row) + " events");
    }
    concurrency(); std::cout << cases.size() - 1 << " voice-slot source cases, " << actions << " actions, FIFO/cancellation/duplicate-release checks match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
