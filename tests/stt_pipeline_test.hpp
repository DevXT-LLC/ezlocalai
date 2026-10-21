#pragma once
#include "stt_api_test.hpp"
inline size_t run_pipeline_case(Environment &env,const Json &scenario,const std::string &label,const fs::path &fixtures = {}) {
    auto probes = native_resource_probes(); probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; }; probes.system_ram = [] { return 16.; }; probes.now = [] { return 100.; };
    auto resources = std::make_shared<ResourceManager>(probes); size_t count = 0;
    {
        const auto preload = [&] { return (env.config.at("preload").get<bool>() || env.config.at("voice_mode").get<bool>()) && !env.config.at("handoff").get<bool>(); };
        auto pool = env.factory(resources); pool.should_preload = preload;
        VoiceSlotOperations voice; voice.should_handoff = [&](const auto &) { return env.config.at("handoff").get<bool>(); };
        voice.unload = [&](const auto &,bool) { env.event("unload"); return "{\"loaded_models\":[\"text\"]}"; };
        voice.restore = [&](const auto &,const auto &,const auto &) { env.event("restore",{{"reload",env.config.value("reload",false)}}); };
        SttService service("fixture-model",env.config.at("capacity"),std::move(pool),std::move(voice),std::make_shared<VoicePermitQueue>(1,true),resources,[&] { return env.config.at("voice_mode").get<bool>(); });
        if (preload()) service.warm();
        for (const auto &action : scenario.at("actions")) {
            if (action.value("recover",false)) { env.config.erase("infer_error"); env.config.erase("load_error"); }
            auto request = action.at("request"); if (action.at("video").get<bool>()) request["return_segments"] = true;
            if (action.contains("audio")) { const auto bytes = read(fixtures/"stt_runtime_assets"/action.at("audio").get<std::string>()); std::string encoded(4*((bytes.size()+2)/3)+1,'\0'); const auto n = EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()),reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size()); expect(n >= 0,"encode pipeline audio"); encoded.resize(n); request["base64_audio"] = encoded; }
            Json result;
            try { result["value"] = Json::parse(service.transcribe(request.dump(),false,action.at("key"),action.value("tts_pipeline",false) ? SttServiceUse::TtsPipeline : action.at("video").get<bool>() ? SttServiceUse::LlmVideo : SttServiceUse::LlmAudio)); }
            catch (const SttError &error) { result["error"] = {{"category",error.category},{"message",error.what()}}; }
            catch (const std::exception &error) { result["error"] = {{"category","RuntimeError"},{"message",error.what()}}; }
            result["events"] = env.events; result["state"] = env.state(service.snapshot(),*resources);
            compare_json(result,scenario.at("outcomes")[count],label + " call " + std::to_string(count)); ++count;
        }
    }
    for (auto &work : env.pending) work(); env.pending.clear(); return count;
}
