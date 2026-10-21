#pragma once
#include "ezlocalai/tts_pool.hpp"
#include "ezlocalai/resource_manager.hpp"
#include "whisper_transcription_test.hpp"
#include <atomic>
#include <future>
#include <thread>
#include <openssl/evp.h>
using namespace transcription_test;
struct Environment {
    Json config, events = Json::array();
    std::map<std::string,int> counts;
    struct Model { TtsPoolHandle handle; std::string device; int closed = 0; bool closed_flag = false; };
    std::map<int,Model> models;
    std::map<TtsPoolModel *,int> ids;
    std::deque<std::function<void()>> pending;
    int next = 0;
    bool preload, handoff, cuda, available;
    Json environment;
    std::shared_ptr<ResourceManager> resources;
    TtsPoolOperations native;
    explicit Environment(Json value) : config(std::move(value)),preload(config.at("preload")),handoff(config.value("handoff",false)),cuda(config.value("cuda",false)),available(config.value("available",true)),environment(config.value("environment",Json::object())) {
        auto probes = native_resource_probes(); probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; };
        probes.system_ram = [] { return 16.; }; probes.now = [this] { event("now"); return 100.; }; resources = std::make_shared<ResourceManager>(probes);
    }
    void event(const std::string &name,Json fields = Json::object()) {
        fields["event"] = name; events.push_back(fields); const auto count = ++counts[name];
        for (const auto &failure : config.value("failures",Json::object()).value(name,Json::array())) if (failure == count) throw TtsError(config.value("failure_category",std::string("RuntimeError")),name + " failure");
    }
    Json id(const TtsPoolHandle &model) const { return model ? Json(ids.at(model.get())) : Json(); }
    TtsPoolOperations operations() {
        auto out = tts_pool_resources({},resources);
        out.environment = [this](const std::string &key)->std::optional<std::string> { event("env",{{"key",key}}); return environment.contains(key)?std::optional<std::string>(environment.at(key).get<std::string>()):std::nullopt; };
        out.available = [this] { return available; };
        out.now = [this] { event("now"); return 100.; };
        out.can_load = [this] { event("can_load",{{"required_vram",4.0}}); return ModelAdmission{config.value("admit",false),config.value("admission_device",std::string("cpu")),config.value("reason",std::string("fixture"))}; };
        out.register_model = [this,call = out.register_model](const auto &name,const auto &device,double vram) { event("register",{{"name",name},{"device",device},{"vram",vram}}); call(name,device,vram); };
        out.unregister_model = [this,call = out.unregister_model] { event("unregister"); call(); };
        out.mark_in_use = [this,call = out.mark_in_use](bool active) { event("mark",{{"active",active}}); call(active); };
        out.load = [this](const std::optional<std::string> &device) {
            const auto key = ++next; event("load",{{"device",device?Json(*device):Json()},{"id",key}}); auto model = std::make_shared<TtsPoolModel>();
            const auto devices = config.value("devices",Json::array({device && !device->empty()?*device:"cpu"})); models[key] = {model,devices[(key - 1) % devices.size()],0,false}; ids[model.get()] = key;
            model->device = [this,key] { event("device",{{"id",key}}); const auto value=models.at(key).device; if(value=="<missing>")throw TtsError("AttributeError","device"); return value; };
            model->closed = [this,key] { event("closed",{{"id",key}}); return models.at(key).closed_flag; };
            model->close = [this,key] { event("close",{{"id",key}}); ++models.at(key).closed; models.at(key).closed_flag=true; }; return model;
        };
        out.should_preload = [this] { event("preload"); return preload; };
        out.handoff_active = [this] { event("handoff"); return handoff; };
        out.cuda_available = [this] { event("cuda"); return cuda; };
        out.collect = [this] { event("collect"); if (native.collect) native.collect(); }; out.synchronize = [this] { event("synchronize"); };
        out.empty_cache = [this] { event("empty_cache"); }; out.ipc_collect = [this] { event("ipc_collect"); };
        out.schedule_cleanup = [this](auto work) { event("schedule"); pending.push_back(std::move(work)); }; return out;
    }
    Json state(const TtsPool &pool) const {
        const auto snapshot = pool.snapshot(); Json out{{"primary",id(snapshot.primary)},{"resident",Json::array()},{"transient",Json::array()},{"available",Json::array()},{"leases",Json::array()},{"pending",pending.size()},{"models",Json::object()},{"resource",nullptr}};
        for (const auto &model : snapshot.resident) out["resident"].push_back(id(model));
        for (const auto &model : snapshot.transient) out["transient"].push_back(id(model));
        for (const auto &model : snapshot.available) out["available"].push_back(id(model));
        for (const auto &lease : snapshot.leases) out["leases"].push_back(Json::array({lease.first,id(lease.second)}));
        for (const auto &model : models) out["models"][std::to_string(model.first)] = {{"device",model.second.device},{"closed",model.second.closed},{"closed_flag",model.second.closed_flag}};
        const auto loaded = resources->loaded_models(); if (!loaded.empty()) { const auto &model = loaded.front(); out["resource"] = {{"name",model.name},{"device",model.device},{"vram",model.vram_gb},{"active",model.active_requests}}; }
        return out;
    }
};
inline size_t run_pool_scenario(Environment &env,TtsPool &pool,const Json &scenario,const std::string &label) {
    uint64_t key = 0; size_t step = 0;
    for (const auto &action : scenario.at("actions")) {
        key = action.value("key",key); Json result = Json::object(); const std::string op = action.at("op");
        try {
            if (op == "warm") pool.warm();
            else if (op == "get") result["value"] = env.id(pool.get(key,action.value("force_cpu",false)));
            else if (op == "name") result["value"] = pool.name();
            else if (op == "create") result["value"] = env.id(pool.create(action.contains("device")?std::optional<std::string>(action.at("device").get<std::string>()):std::nullopt,action.value("force_cpu",false)));
            else if (op == "load") result["value"] = env.id(pool.load_instance(action.value("force_cpu",false)));
            else if (op == "transfer") pool.transfer(action.at("owner"),key);
            else if (op == "available") env.available = action.at("value");
            else if (op == "env") { const std::string name=action.at("name"); if(action.at("value").is_null())env.environment.erase(name);else env.environment[name]=action.at("value"); }
            else if (op == "closed") { const int id=action.at("id");if(env.models.count(id))env.models.at(id).closed_flag=action.at("value"); }
            else if (op == "destroy") pool.destroy(key,action.value("async",true),action.value("force",false));
            else if (op == "register") pool.register_resources();
            else if (op == "mark") env.operations().mark_in_use(action.at("active"));
            else if (op == "preload") env.preload = action.at("value");
            else if (op == "handoff") env.handoff = action.at("value");
            else if (op == "cuda") env.cuda = action.at("value");
            else if (op == "device") { const int id = action.at("id"); if (env.models.count(id)) env.models.at(id).device = action.at("value"); }
            else if (op == "drain") { auto count = std::min(action.value("count",env.pending.size()),env.pending.size()); while (count--) { auto work = std::move(env.pending.front()); env.pending.pop_front(); work(); } }
            else throw std::runtime_error("Unknown pool test action " + op);
        } catch (const TtsError &error) { result["error"] = {{"category",error.category},{"message",error.what()}}; }
        catch (const std::exception &error) { result["error"] = {{"category","RuntimeError"},{"message",error.what()}}; }
        result["state"] = env.state(pool); compare_json(result,scenario.at("outcomes")[step],label + " action " + std::to_string(step)); ++step;
    }
    compare_json(env.events,scenario.at("events"),label + " event order"); return step;
}
