#pragma once
#include "ezlocalai/stt_pool.hpp"
#include "ezlocalai/resource_manager.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "whisper_transcription_test.hpp"
#include <atomic>
#include <future>
#include <thread>
#include <openssl/evp.h>
using namespace transcription_test;
struct Environment {
    Json config, events = Json::array();
    std::map<std::string,int> counts;
    struct Model { SttPoolHandle handle; std::string device; int closed = 0; };
    std::map<int,Model> models;
    std::map<SttPoolModel *,int> ids;
    std::deque<std::function<void()>> pending;
    int next = 0;
    bool preload, handoff, cuda;
    std::shared_ptr<ResourceManager> resources;
    SttPoolOperations native;
    explicit Environment(Json value) : config(std::move(value)),preload(config.at("preload")),handoff(config.value("handoff",false)),cuda(config.value("cuda",false)) {
        auto probes = native_resource_probes(); probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; };
        probes.system_ram = [] { return 16.; }; probes.now = [] { return 100.; }; resources = std::make_shared<ResourceManager>(probes);
    }
    void event(const std::string &name,Json fields = Json::object()) {
        fields["event"] = name; events.push_back(fields); const auto count = ++counts[name];
        for (const auto &failure : config.value("failures",Json::object()).value(name,Json::array())) if (failure == count) throw std::runtime_error(name + " failure");
    }
    Json id(const SttPoolHandle &model) const { return model ? Json(ids.at(model.get())) : Json(); }
    SttPoolOperations operations() {
        auto out = stt_pool_resources({},resources);
        out.can_load = [this] { event("can_load"); };
        out.register_model = [this,call = out.register_model](const auto &name,const auto &device,double vram) { event("register",{{"name",name},{"device",device},{"vram",vram}}); call(name,device,vram); };
        out.unregister_model = [this,call = out.unregister_model] { event("unregister"); call(); };
        out.mark_in_use = [this,call = out.mark_in_use](bool active) { event("mark",{{"active",active}}); call(active); };
        out.load = [this](const std::string &name) {
            const auto key = ++next; event("load",{{"name",name},{"id",key}}); auto model = native.load ? native.load(name) : std::make_shared<SttPoolModel>();
            const auto devices = config.value("devices",Json::array({"cpu"})); models[key] = {model,devices[(key - 1) % devices.size()],0}; ids[model.get()] = key;
            if (native.load) expect(model->device() == models[key].device,"native pool device");
            model->device = [this,key] { return models.at(key).device; };
            model->close = [this,key,close = model->close] { event("close",{{"id",key}}); if (close) close(); ++models.at(key).closed; }; return model;
        };
        out.should_preload = [this] { event("preload"); return preload; };
        out.handoff_active = [this] { event("handoff"); return handoff; };
        out.cuda_available = [this] { event("cuda"); return cuda; };
        out.collect = [this] { event("collect"); if (native.collect) native.collect(); }; out.synchronize = [this] { event("synchronize"); };
        out.empty_cache = [this] { event("empty_cache"); }; out.ipc_collect = [this] { event("ipc_collect"); };
        out.schedule_cleanup = [this](auto work) { event("schedule"); pending.push_back(std::move(work)); }; return out;
    }
    Json state(const SttPool &pool) const {
        const auto snapshot = pool.snapshot(); Json out{{"primary",id(snapshot.primary)},{"resident",Json::array()},{"transient",Json::array()},{"available",Json::array()},{"leases",Json::array()},{"pending",pending.size()},{"models",Json::object()},{"resource",nullptr}};
        for (const auto &model : snapshot.resident) out["resident"].push_back(id(model));
        for (const auto &model : snapshot.transient) out["transient"].push_back(id(model));
        for (const auto &model : snapshot.available) out["available"].push_back(id(model));
        for (const auto &lease : snapshot.leases) out["leases"].push_back(Json::array({lease.first,id(lease.second)}));
        for (const auto &model : models) out["models"][std::to_string(model.first)] = {{"device",model.second.device},{"closed",model.second.closed}};
        const auto loaded = resources->loaded_models(); if (!loaded.empty()) { const auto &model = loaded.front(); out["resource"] = {{"name",model.name},{"device",model.device},{"vram",model.vram_gb},{"active",model.active_requests}}; }
        return out;
    }
};
inline size_t run_pool_scenario(Environment &env,SttPool &pool,const Json &scenario,const std::string &label,const fs::path &fixtures = {}) {
    uint64_t key = 0; size_t step = 0;
    for (const auto &action : scenario.at("actions")) {
        key = action.value("key",key); Json result = Json::object(); const std::string op = action.at("op");
        try {
            if (op == "warm") pool.warm();
            else if (op == "get") result["value"] = env.id(pool.get(key,action.value("force_cpu",false)));
            else if (op == "destroy") pool.destroy(key,action.value("async",true),action.value("force",false));
            else if (op == "register") pool.register_resources();
            else if (op == "mark") env.operations().mark_in_use(action.at("active"));
            else if (op == "preload") env.preload = action.at("value");
            else if (op == "handoff") env.handoff = action.at("value");
            else if (op == "cuda") env.cuda = action.at("value");
            else if (op == "device") { const int id = action.at("id"); if (env.models.count(id)) env.models.at(id).device = action.at("value"); }
            else if (op == "drain") { auto count = std::min(action.value("count",env.pending.size()),env.pending.size()); while (count--) { auto work = std::move(env.pending.front()); env.pending.pop_front(); work(); } }
            else if (op == "transcribe") {
                const auto bytes = read(fixtures/"stt_runtime_assets"/action.at("audio").get<std::string>()); std::string encoded(4 * ((bytes.size() + 2) / 3) + 1,'\0');
                const auto size = EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()),reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size()); expect(size >= 0,"encode pool audio"); encoded.resize(size);
                auto request = action.at("request"); request["base64_audio"] = encoded;
                result["value"] = Json::parse(env.models.at(action.at("id")).handle->transcribe(request.dump()));
            } else throw std::runtime_error("Unknown pool test action " + op);
        } catch (const SttError &error) { result["error"] = {{"category",error.category},{"message",error.what()}}; }
        catch (const std::exception &error) { result["error"] = {{"category","RuntimeError"},{"message",error.what()}}; }
        result["state"] = env.state(pool); compare_json(result,scenario.at("outcomes")[step],label + " action " + std::to_string(step)); ++step;
    }
    compare_json(env.events,scenario.at("events"),label + " event order"); return step;
}
