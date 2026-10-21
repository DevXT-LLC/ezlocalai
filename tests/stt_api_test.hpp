#pragma once
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
using namespace transcription_test;
std::string decode(const std::string &input) { std::string out(input.size(),'\0'); auto n = EVP_DecodeBlock(reinterpret_cast<unsigned char *>(out.data()),reinterpret_cast<const unsigned char *>(input.data()),input.size()); expect(n >= 0,"decode multipart fixture"); if (!input.empty() && input.back() == '=') --n; if (input.size() > 1 && input[input.size()-2] == '=') --n; out.resize(n); return out; }
struct Environment {
    std::mutex chat_mutex; Json chats = Json::array();
    Json config,events = Json::array(); SttPoolOperations native; bool real = false; int next = 0; std::map<int,int> closed; std::map<SttPoolModel *,int> ids; std::vector<SttPoolHandle> models; std::vector<std::function<void()>> pending;
    void event(const std::string &name,Json fields = Json::object()) { fields["event"] = name; events.push_back(fields); }
    SttPoolOperations factory(std::shared_ptr<ResourceManager> resources) {
        auto ops = stt_pool_resources({},resources);
        ops.load = [&,resources](const std::string &name) {
            const auto id = ++next; event("load",{{"id",id},{"name",name}}); if (config.value("load_error",false)) throw SttError("ValueError","fixture load failure");
            auto model = native.load ? native.load(name) : std::make_shared<SttPoolModel>(); ids[model.get()] = id; models.push_back(model); closed[id] = 0; model->device = [] { return "cpu"; };
            model->close = [&,id,close = model->close] { event("close",{{"id",id}}); if (close) close(); ++closed[id]; };
            model->transcribe = [&,id,resources,infer = model->transcribe](const auto &encoded) {
                if (config.value("record_usage",false)) { int64_t active = 0; for (const auto &model : resources->loaded_models()) if (model.type == ModelType::Stt) active = model.active_requests; event("usage",{{"active",active}}); }
                const auto request = Json::parse(encoded); auto recorded = request; if (real) recorded["base64_audio"] = "<AUDIO>"; event("infer",{{"id",id},{"request",recorded}}); if (infer) return infer(encoded);
                if (config.contains("infer_error")) throw SttError(config.at("infer_error"),"fixture inference failure");
                if (config.contains("result")) return config.at("result").dump();
                if (!request.value("return_segments",false)) return Json(" hello world ").dump();
                return std::string(R"({"text":"hello world","language":"en","segments":[{"id":0,"start":0.1,"end":1.234,"text":" hello","words":[{"word":" hello","start":0.1,"end":1.234,"probability":0.9}]},{"id":1,"start":1.235,"end":3.0,"text":" world","speaker":"SPEAKER_00"}]})");
            }; return model;
        };
        ops.schedule_cleanup = [&](auto work) { pending.push_back(std::move(work)); }; return ops;
    }
    Json state(const NativeServer &server,const ResourceManager &resources) { return state(*server.stt_pool_snapshot(),resources); }
    Json state(const SttPoolSnapshot &snapshot,const ResourceManager &resources) {
        Json out{{"resident",Json::array()},{"transient",Json::array()},{"available",Json::array()},{"leases",Json::array()},{"pending",pending.size()},{"closed",Json::object()},{"resource",nullptr}};
        for (const auto &model : snapshot.resident) out["resident"].push_back(ids.at(model.get())); for (const auto &model : snapshot.transient) out["transient"].push_back(ids.at(model.get()));
        for (const auto &model : snapshot.available) out["available"].push_back(ids.at(model.get())); for (const auto &lease : snapshot.leases) out["leases"].push_back(ids.at(lease.second.get()));
        for (const auto &entry : closed) out["closed"][std::to_string(entry.first)] = entry.second;
        for (const auto &model : resources.loaded_models()) if (model.type == ModelType::Stt) out["resource"] = {{"name",model.name},{"device",model.device},{"vram",model.vram_gb},{"active",model.active_requests}};
        return out;
    }
};
struct Backend : InferenceBackend {
    Environment &env; explicit Backend(Environment &value) : env(value) {}
    std::string chat(const std::string &body) override {
        std::lock_guard<std::mutex> lock(env.chat_mutex); env.chats.push_back(Json::parse(body));
        if (env.config.value("chat_error",false)) throw std::runtime_error("fixture chat failure");
        return env.config.value("chat_response",Json{{"choices",Json::array({Json{{"message",Json{{"content","[1] translated one\n[2] translated two"}}}}})}}).dump();
    } std::string completion(const std::string &) override { return "{}"; } std::string embedding(const std::string &) override { return "{}"; }
    std::function<void(bool)> unload_llms_for_media() override { env.event("unload"); return [this](bool reload) { env.event("restore",{{"reload",reload}}); }; }
};

inline size_t run_api_case(Environment &env,const Json &scenario,const std::string &label,const fs::path &fixtures = {}) {
 size_t count = 0;
 {
        for (const auto &entry : environment_defaults) setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
        setenv("EZLOCALAI_API_KEY","",1); setenv("DEFAULT_MODEL","fixture-text",1); setenv("WHISPER_MODEL","fixture-model",1); setenv("VOICE_SERVER",env.config.at("voice_mode").get<bool>() ? "true" : "",1);
        setenv("STT_N_PARALLEL",std::to_string(env.config.at("capacity").get<int>()).c_str(),1); setenv("LAZY_LOAD_VOICE",env.config.at("preload").get<bool>() ? "false" : "true",1);
        setenv("STT_UNLOAD_LLM_DURING_GENERATION",env.config.at("handoff").get<bool>() ? "true" : "false",1); setenv("STT_RELOAD_LLM_AFTER_GENERATION",env.config.value("reload",false) ? "true" : "false",1);
        auto probes = native_resource_probes(); probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; }; probes.system_ram = [] { return 16.; }; probes.now = [] { return 100.; };
        auto resources = std::make_shared<ResourceManager>(probes); ezl_env_config config{}; ezl_load_config(&config);
        NativeServer server(config,std::make_unique<Backend>(env),resources,[&](auto resources) { return env.factory(resources); });
        for (const auto &row : scenario.at("outcomes")) {
            auto body = decode(row.at("body")); if (row.contains("audio")) { const auto at = body.find("<AUDIO>"); expect(at != std::string::npos,"audio placeholder"); body.replace(at,7,read(fixtures/"stt_runtime_assets"/row.at("audio").get<std::string>())); }
            const auto response = server.handle_request("POST",row.at("path"),"",body,row.at("content_type")); const auto request_label = label + " request " + std::to_string(count);
            expect(response.status == row.at("status"),request_label + " status " + std::to_string(response.status) + " " + response.body); { // Fixtures record the handler's media_type; Starlette adds the charset on the wire.
                std::string mime = row.at("mime"); if (mime.rfind("text/", 0) == 0 && mime.find("charset=") == std::string::npos) mime += "; charset=utf-8";
                expect(response.content_type == mime,request_label + " MIME " + response.content_type); }
            if (response.content_type == "application/json") compare_json(Json::parse(response.body),Json::parse(row.at("response").get<std::string>()),request_label + " body"); else compare_json(response.body,row.at("response"),request_label + " body");
            compare_json(env.events,row.at("events"),request_label + " lifecycle"); compare_json(env.state(server,*resources),row.at("state"),request_label + " pool state"); ++count;
            if (row.contains("chats")) {
                auto actual = env.chats.get<std::vector<Json>>(),expected = row.at("chats").get<std::vector<Json>>();
                const auto less = [](const auto &a,const auto &b) { return a.dump() < b.dump(); }; std::sort(actual.begin(),actual.end(),less); std::sort(expected.begin(),expected.end(),less);
                compare_json(Json(actual),Json(expected),request_label + " translation requests");
            }
        }
 }
 for (auto &work : env.pending) work(); env.pending.clear(); return count;
}
