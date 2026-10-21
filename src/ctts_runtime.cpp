#include "ezlocalai/ctts_runtime.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_slots.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <mutex>
namespace ezlocalai {
namespace {
using Json=nlohmann::ordered_json;
bool truth(const Json &v) { if(v.is_null()) return false; if(v.is_boolean()) return v.get<bool>(); if(v.is_number()) return v.get<double>()!=0; if(v.is_string()) return !v.get_ref<const std::string &>().empty(); return !v.empty(); }
std::string type(const Json &v) { return v.is_null()?"NoneType":v.is_boolean()?"bool":v.is_string()?"str":v.is_array()?"list":v.is_object()?"dict":v.is_number_float()?"float":"int"; }
TtsError current_error() { try { throw; } catch(const TtsError &e) { return e; } catch(const std::exception &e) { return {"RuntimeError",e.what()}; } }
bool runtime(const TtsError &e) { return e.runtime_family || e.category=="RuntimeError" || e.category=="OutOfMemoryError" || e.category=="LibsndfileError" || e.category=="SoundFileRuntimeError" || e.category=="NotImplementedError" || e.category=="RecursionError"; }
std::string env(const CttsOperations &ops,const std::string &key,const std::string &fallback) { const auto value=ops.environment(key); return value?*value:fallback; }
std::string env_int(const CttsOperations &ops,const std::string &key,const std::string &fallback) { const auto raw=ops.environment(key); if(!raw) return fallback; const auto value=parse_configuration_integer(*raw); return value?*value:fallback; }
bool env_bool(const CttsOperations &ops,const std::string &key,bool fallback) { const auto raw=ops.environment(key); if(!raw) return fallback; const auto text=llm_unicode_lower(strip_config_value(*raw)); return text=="1" || text=="true" || text=="yes" || text=="on"; }
std::string maximum(const std::string &a,const std::string &b) { return configuration_integer_less(a,b)?b:a; }
std::string minimum(const std::string &a,const std::string &b) { return configuration_integer_less(a,b)?a:b; }
Json generation(const CttsOperations &ops) {
    const auto raw=strip_config_value(env(ops,"QWEN_TTS_GENERATE_KWARGS","")); Json kwargs=Json::object();
    if(!raw.empty()) { try { auto parsed=Json::parse(raw); if(parsed.is_object()) kwargs=std::move(parsed); } catch(const Json::exception &) {} }
    const auto max_tokens=env_int(ops,"QWEN_TTS_MAX_NEW_TOKENS","320"); if(configuration_integer_less("0",max_tokens) && !kwargs.contains("max_new_tokens") && !kwargs.contains("max_length")) kwargs["max_new_tokens"]=Json::parse(max_tokens); return kwargs;
}
CttsSettings settings(const CttsOperations &ops) {
    CttsSettings s; s.model_id=resolve_tts_model_id(env(ops,"QWEN_TTS_MODEL","Mouserat/qwen3-tts-0.6b-base-gguf"));
    s.max_chunk_chars=maximum("1",env_int(ops,"QWEN_TTS_MAX_CHUNK_CHARS","350"));
    s.stream_chunk_target_chars=minimum(s.max_chunk_chars,maximum("1",env_int(ops,"QWEN_TTS_STREAM_CHUNK_CHARS","280")));
    s.stream_first_chunk_target_chars=minimum(s.stream_chunk_target_chars,maximum("1",env_int(ops,"QWEN_TTS_STREAM_FIRST_CHUNK_CHARS","120")));
    s.stream_min_first_chunk_chars=minimum(s.stream_first_chunk_target_chars,maximum("1",env_int(ops,"QWEN_TTS_STREAM_MIN_FIRST_CHUNK_CHARS","50")));
    s.stream_write_bytes=maximum("1024",env_int(ops,"QWEN_TTS_STREAM_WRITE_BYTES","16384"));
    const auto raw=ops.environment("QWEN_TTS_STREAM_FRAME_DRAIN_SECONDS"); double drain=0; if(raw) parse_configuration_float(*raw,&drain); s.stream_frame_drain_seconds=std::max(0.0,drain);
    s.stream_flush_silence_ms=maximum("0",env_int(ops,"QWEN_TTS_STREAM_FLUSH_SILENCE_MS","0")); s.non_streaming_mode=env_bool(ops,"QWEN_TTS_NON_STREAMING_MODE",true); s.allow_cpu_fallback=env_bool(ops,"QWEN_TTS_ALLOW_CPU_FALLBACK",true); s.generation_kwargs=generation(ops).dump(); return s;
}
}
std::string CttsSettings::json() const {
    Json drain=stream_frame_drain_seconds; if(std::isinf(stream_frame_drain_seconds)) drain=Json{{"__float__","inf"}};
    return Json{{"model_id",model_id},{"max_chunk_chars",max_chunk_chars},{"stream_chunk_target_chars",stream_chunk_target_chars},{"stream_first_chunk_target_chars",stream_first_chunk_target_chars},{"stream_min_first_chunk_chars",stream_min_first_chunk_chars},{"stream_write_bytes",stream_write_bytes},{"stream_frame_drain_seconds",drain},{"stream_flush_silence_ms",stream_flush_silence_ms},{"non_streaming_mode",non_streaming_mode},{"default_x_vector_only",true},{"allow_cpu_fallback",allow_cpu_fallback},{"generation_kwargs",Json::parse(generation_kwargs)}}.dump();
}
CttsOperations ctts_llama_operations(CttsOperations ops) {
    if(!ops.environment) ops.environment=[](const auto &key)->std::optional<std::string> { const auto *value=std::getenv(key.c_str()); return value?std::optional<std::string>(value):std::nullopt; };
    if(!ops.cuda_available) ops.cuda_available=[] { return false; };
    if(!ops.memory_info) ops.memory_info=[](const auto &)->std::pair<double,double> { throw TtsError("RuntimeError","CUDA memory information unavailable"); };
    if(!ops.total_bytes) ops.total_bytes=[](const auto &)->double { throw TtsError("RuntimeError","CUDA properties unavailable"); };
    if(!ops.allocated_bytes) ops.allocated_bytes=[](const auto &) { return 0.0; };
    for(auto *action:{&ops.collect,&ops.empty_cache,&ops.synchronize,&ops.ipc_collect}) if(!*action) *action=[] {};
    if(!ops.inference_mode) ops.inference_mode=[] { return [](const std::string &) {}; };
    if(!ops.load) ops.load=[](const auto &device,const auto &kwargs) { const auto model=std::make_shared<LlamaTts>(llama_tts_launch(Json(device).dump(),kwargs,native_llama_tts_platform())); auto out=std::make_shared<CttsBackend>(); out->generate=[model](const auto &request) { auto audio=model->generate(request); const auto rate=audio.rate; return CttsGenerated{{std::move(audio)},rate}; }; out->stream=[model](const auto &request) { return model->stream_iterator(request); }; out->close=[model] { model->close(); }; return out; };
    return ops;
}
struct CttsRuntime::Impl {
    CttsOperations ops; CttsSettings config; mutable std::mutex mutex; std::string device; bool closed=false; std::shared_ptr<CttsBackend> model;
    explicit Impl(CttsOperations value):ops(ctts_llama_operations(std::move(value))) {}
    double available_vram(const std::string &index) { if(ops.cuda_available()) { try { return ops.memory_info(index).first/(1024*1024); } catch(const std::exception &) { try { const auto total=ops.total_bytes(index); const auto allocated=ops.allocated_bytes(index); return (total-allocated)/(1024*1024); } catch(const std::exception &) { return 0; } } } return 0; }
    std::string select(const Json &value) {
        const auto requested=llm_unicode_lower(strip_config_value(truth(value)?tts_python_string(value.dump()):"")); std::string index="0";
        if(requested.rfind("cuda:",0)==0) { const auto parsed=parse_configuration_integer(requested.substr(5)); if(parsed) index=*parsed; }
        const auto min_vram=env_int(ops,"QWEN_TTS_MIN_VRAM_MB","3500"); const auto available=available_vram(index); if(requested=="cpu") return "cpu";
        if(ops.cuda_available() && requested.rfind("cuda",0)==0) return requested;
        // Python compares the memory float with an unbounded integer threshold.
        const auto meets_threshold=[&] { if(std::isnan(available)) return false; if(std::isinf(available)) return available>0; char decimal[512]; std::snprintf(decimal,sizeof(decimal),"%.0f",std::floor(available)); const std::string whole=available==0?"0":decimal; return !configuration_integer_less(whole,min_vram); };
        if(ops.cuda_available() && requested.empty() && meets_threshold()) return "cuda:0";
        if(ops.cuda_available() && !config.allow_cpu_fallback) return "cuda:0";
        if(ops.cuda_available()) return "cpu";
        if(!config.allow_cpu_fallback) throw TtsError("RuntimeError","[QTTS] CUDA is not available and QWEN_TTS_ALLOW_CPU_FALLBACK is not enabled. Set QWEN_TTS_ALLOW_CPU_FALLBACK=true or request device='cpu' to try CPU generation."); return "cpu";
    }
};
CttsRuntime::CttsRuntime(const std::string &requested,CttsOperations ops):impl_(std::make_unique<Impl>(std::move(ops))) {
    impl_->config=ezlocalai::settings(impl_->ops); impl_->device=impl_->select(Json::parse(requested));
    try { impl_->model=impl_->ops.load(impl_->device,impl_->config.generation_kwargs); }
    catch(...) { const auto error=current_error(); if(!runtime(error) || impl_->device.rfind("cuda",0)!=0 || !impl_->config.allow_cpu_fallback) throw; impl_->device="cpu"; impl_->model=impl_->ops.load("cpu",impl_->config.generation_kwargs); }
}
CttsRuntime::~CttsRuntime()=default;
const CttsSettings &CttsRuntime::settings() const { return impl_->config; }
std::string CttsRuntime::device() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->device; }
bool CttsRuntime::closed() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->closed; }
std::shared_ptr<CttsBackend> CttsRuntime::backend() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->model; }
std::string CttsRuntime::warmup_generation_kwargs() const {
    auto kwargs=Json::parse(impl_->config.generation_kwargs); const auto max_tokens=env_int(impl_->ops,"QWEN_TTS_WARMUP_MAX_NEW_TOKENS","32"); if(!configuration_integer_less("0",max_tokens)) return kwargs.dump();
    if(kwargs.contains("max_new_tokens") && (kwargs["max_new_tokens"].is_number_integer() || kwargs["max_new_tokens"].is_boolean())) { const auto value=tts_python_integer(kwargs["max_new_tokens"].dump()); if(configuration_integer_less("0",value)) { if(configuration_integer_less(max_tokens,value)) kwargs["max_new_tokens"]=Json::parse(max_tokens); return kwargs.dump(); } }
    kwargs["max_new_tokens"]=Json::parse(max_tokens); return kwargs.dump();
}
void CttsRuntime::release_generation_memory() { impl_->ops.collect(); if(device().rfind("cuda",0)==0 && impl_->ops.cuda_available()) { try { impl_->ops.empty_cache(); impl_->ops.ipc_collect(); } catch(const std::exception &) {} } }
void CttsRuntime::close() {
    std::shared_ptr<CttsBackend> model; { std::lock_guard<std::mutex> lock(impl_->mutex); impl_->closed=true; model=std::move(impl_->model); }
    if(model && model->close) model->close(); model.reset(); impl_->ops.collect(); if(impl_->ops.cuda_available()) { try { impl_->ops.synchronize(); } catch(const std::exception &) {} impl_->ops.empty_cache(); try { impl_->ops.ipc_collect(); } catch(const std::exception &) {} }
}
void CttsRuntime::reload_on_cpu() {
    auto model=backend(); if(model && model->close) model->close(); release_generation_memory(); { std::lock_guard<std::mutex> lock(impl_->mutex); impl_->device="cpu"; } auto loaded=impl_->ops.load("cpu",impl_->config.generation_kwargs); { std::lock_guard<std::mutex> lock(impl_->mutex); impl_->model=std::move(loaded); }
}
TtsAudio CttsRuntime::generate_sample(const std::string &encoded) {
    const auto input=Json::parse(encoded); const auto audio_path=input.value("audio_path",Json()); if(!truth(audio_path)) throw TtsError("ValueError","[QTTS] Qwen Base TTS requires a reference voice wav");
    auto ref=input.value("ref_text",Json()),x_vector=input.value("x_vector_only",Json()); if(x_vector.is_null()) { ref=nullptr; x_vector=true; }
    auto kwargs=input.value("generation_kwargs",Json()); if(kwargs.is_null()) kwargs=Json::parse(impl_->config.generation_kwargs);
    const auto invoke=[&]() {
        auto finish=impl_->ops.inference_mode(); CttsGenerated result;
        try {
            auto model=backend(); if(!model) throw TtsError("AttributeError","'NoneType' object has no attribute 'generate_voice_clone'");
            Json request{{"text",input.at("text")},{"language",input.at("qwen_language")},{"ref_audio",audio_path},{"ref_text",ref},{"x_vector_only_mode",x_vector},{"non_streaming_mode",impl_->config.non_streaming_mode}};
            if(!kwargs.is_object()) throw TtsError("TypeError",model->generate_name+"() argument after ** must be a mapping, not "+type(kwargs));
            for(const auto &entry:kwargs.items()) { if(request.contains(entry.key())) throw TtsError("TypeError",model->generate_name+"() got multiple values for keyword argument '"+entry.key()+"'"); request[entry.key()]=entry.value(); }
            result=model->generate(request.dump());
        } catch(...) { const auto error=current_error(); finish(error.category); throw; }
        finish(""); if(result.audio.empty()) throw TtsError("IndexError","list index out of range"); auto audio=std::move(result.audio[0]); audio.rate=tts_python_integer(result.rate_json); return audio;
    };
    std::exception_ptr failure; TtsAudio result;
    try { try { result=invoke(); } catch(...) { const auto error=current_error(); if(!runtime(error) || closed()) throw; const auto message=llm_unicode_lower(error.what()); if(message.find("out of memory")==std::string::npos && message.find("cuda")==std::string::npos && message.find("cudnn")==std::string::npos) throw;
            if(!impl_->config.allow_cpu_fallback) throw TtsError("RuntimeError","[QTTS] GPU generation failed and QWEN_TTS_ALLOW_CPU_FALLBACK is not enabled. Reduce TTS_N_PARALLEL, free VRAM, or set QWEN_TTS_ALLOW_CPU_FALLBACK=true to try CPU generation."); reload_on_cpu(); result=invoke(); }
    } catch(...) { failure=std::current_exception(); }
    release_generation_memory(); if(failure) std::rethrow_exception(failure); return result;
}
}
