#pragma once
#include "ezlocalai/tts_service.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
struct Environment {
    TtsPoolOperations native;
    Json config,events=Json::array();std::map<std::string,int> counts;
    struct Model {TtsPoolHandle handle;int closed=0;bool closed_flag=false;};
    struct Audio {size_t position=0;int closed=0;};
    std::map<int,Model> models;std::map<TtsPoolModel *,int> ids;std::map<int,std::shared_ptr<Audio>> streams;
    std::vector<std::function<void()>> pending;int next=0,stream_id=0,handoff=0,shared=1,permit;bool unavailable=false,mode;
    std::shared_ptr<ResourceManager> resources;
    explicit Environment(Json config):config(std::move(config)),permit(std::max(1,this->config.value("capacity",2))),mode(this->config.value("voice_mode",false)){
        auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 16.;};probes.now=[]{return 100.;};resources=std::make_shared<ResourceManager>(probes);
    }
    void event(const std::string &name,Json fields=Json::object()){
        fields["event"]=name;events.push_back(fields);const auto count=++counts[name];
        for(const auto &failure:config.value("failures",Json::array()))if(failure.at("event")==name&&failure.value("at",1)==count)throw TtsError(failure.value("category",std::string("RuntimeError")),failure.value("message",name+" failure"));
    }
    std::optional<std::string> environment(const std::string &key){if(key=="QWEN_TTS_ENABLE_GTTS_FALLBACK")event("fallback.env");const auto env=config.value("environment",Json::object());return env.contains(key)?std::optional<std::string>(env.at(key).get<std::string>()):std::nullopt;}
    struct Stream:CttsByteStream {
        Environment &env;int id;std::shared_ptr<Audio> audio;std::unique_ptr<CttsByteStream> native;
        Stream(Environment &env,int id,std::shared_ptr<Audio> audio,std::unique_ptr<CttsByteStream> native={}):env(env),id(id),audio(std::move(audio)),native(std::move(native)){}
        std::optional<std::string> next()override{env.event("next",{{"id",id}});if(native){auto result=native->next();if(result)++audio->position;return result;}const auto packets=env.config.value("packets",Json::array({"header","pcm","end"}));if(audio->position==packets.size())return {};const auto value=packets[audio->position++].get<std::string>();return env.config.value("hex_packets",false)?unhex(value):value;}
        void close()override{env.event("stream.close",{{"id",id}});if(native)native->close();++audio->closed;}
        void request_cancel()override{if(native)native->request_cancel();}
    };
    TtsPoolOperations pool(){
        auto ops=tts_pool_resources({},resources);ops.environment=[&](const auto &key){return environment(key);};ops.available=[&]{return config.value("available",true);};ops.should_preload=[&]{return config.value("preload",false);};
        ops.can_load=[&]{event("admission");return ModelAdmission{true,"cpu","fixture"};};
        ops.mark_in_use=[&,mark=ops.mark_in_use](bool active){event("mark",{{"active",active}});mark(active);};
        ops.register_model=[&,call=ops.register_model](const auto &name,const auto &device,double vram){event("register");call(name,device,vram);};ops.unregister_model=[&,call=ops.unregister_model]{event("unregister");call();};
        ops.collect=[&]{event("collect");};ops.schedule_cleanup=[&](auto work){event("schedule");pending.push_back(std::move(work));};
        ops.load=[&](const std::optional<std::string> &device){
            const auto id=++next;event("load",{{"id",id},{"device",device?Json(*device):Json()}});auto model=native.load?native.load(device):std::make_shared<TtsPoolModel>();models[id]={model,0,false};ids[model.get()]=id;
            model->device=[]{return "cpu";};model->closed=[&,id,closed=model->closed]{return closed?closed():models.at(id).closed_flag;};model->close=[&,id,close=model->close]{event("close",{{"id",id}});if(close)close();++models.at(id).closed;models.at(id).closed_flag=true;};
            model->generate=[&,id,generate=model->generate](const auto &request){event("generate",{{"id",id},{"request",Json::parse(request)}});if(generate)return generate(request);return config.value("audio",std::string("UklGRgAB"));};
            model->stream=[&,id,stream=model->stream](const auto &request)->std::unique_ptr<CttsByteStream>{event("stream",{{"id",id},{"request",Json::parse(request)}});const auto key=++stream_id;auto audio=std::make_shared<Audio>();streams[key]=audio;return std::make_unique<Stream>(*this,key,audio,stream?stream(request):nullptr);};return model;
        };return ops;
    }
    VoiceSlotOperations voice(){
        VoiceSlotOperations ops;ops.should_handoff=[&](const auto &){event("handoff");return config.value("handoff",false);};
        ops.acquire_shared=[&](const auto &){event("shared.acquire");if(!shared)throw TtsError("RuntimeError","fixture blocked shared");--shared;};ops.release_shared=[&]{event("shared.release");++shared;};
        ops.acquire_permit=[&](const auto &){event("permit.acquire");if(!permit)throw TtsError("RuntimeError","fixture blocked permit");--permit;};ops.release_permit=[&]{event("permit.release");++permit;};
        ops.mark_handoff=[&](const auto &,bool active){event("handoff.mark",{{"active",active}});handoff=std::max(0,handoff+(active?1:-1));};ops.set_unavailable=[&](bool value){unavailable=value;};
        ops.wait_for_idle=[&](const auto &,const auto &){event("idle");};ops.unload=[&](const auto &,bool){event("unload");return "{\"loaded\":[\"llm\"]}";};
        ops.restore=[&](const auto &,const auto &state,const auto &reload){event("restore",{{"state",Json::parse(state)},{"reload",reload}});};return ops;
    }
    TtsRequestOperations request(){TtsRequestOperations ops;ops.environment=[&](const auto &key){return environment(key);};ops.voice_server_mode=[&]{event("voice_mode");return mode;};ops.mark_in_use=pool().mark_in_use;ops.fallback_generate=[&](const auto &request){const auto data=Json::parse(request);event("fallback",{{"text",data.at("text")},{"language",data.at("language")}});return "fallback WAV";};return ops;}
    Json id(const TtsPoolHandle &model)const{return model?Json(ids.at(model.get())):Json();}
    Json state(const TtsService &service)const{
        const auto pool=service.snapshot();Json out{{"primary",id(pool.primary)},{"resident",Json::array()},{"transient",Json::array()},{"available",Json::array()},{"leases",Json::array()},{"pending",pending.size()},{"models",Json::object()},{"streams",Json::object()},{"resource",nullptr},{"permit",permit},{"shared",shared},{"handoff",handoff},{"unavailable",unavailable},{"stacks",Json::array()}};
        for(const auto &m:pool.resident)out["resident"].push_back(id(m));for(const auto &m:pool.transient)out["transient"].push_back(id(m));for(const auto &m:pool.available)out["available"].push_back(id(m));for(const auto &l:pool.leases)out["leases"].push_back(Json::array({l.first,id(l.second)}));
        for(const auto &[id,model]:models)out["models"][std::to_string(id)]={{"closed",model.closed},{"closed_flag",model.handle->closed()}};for(const auto &[id,audio]:streams)out["streams"][std::to_string(id)]={{"position",audio->position},{"closed",audio->closed}};
        for(const auto &resource:resources->loaded_models())if(resource.type==ModelType::Tts)out["resource"]={{"active",resource.active_requests},{"device",resource.device},{"vram",resource.vram_gb}};
        for(const auto &[key,stack]:service.slot_stacks())out["stacks"].push_back(Json::array({key,stack.size()}));return out;
    }
};
std::string hex(const std::string &value){std::string out;for(unsigned char c:value){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;}
inline size_t run_service_scenario(Environment &env,const Json &scenario,const std::string &label){
    const auto serialize=[&](const std::string &value)->Json{return env.config.value("real",false)?Json{{"size",value.size()},{"sha256",hash(value)}}:Json(hex(value));};
    TtsService service(env.config.value("capacity",2),env.pool(),env.voice(),std::make_shared<VoicePermitQueue>(1,true),env.resources,env.request());std::unique_ptr<TtsServiceStream> stream;uint64_t key=0;size_t step=0,actions=0;
        for(const auto &action:scenario.at("actions")){key=action.value("key",key);const std::string op=action.at("op");const auto request=action.value("request",Json{{"text","Hello."},{"voice","default"},{"language","auto"}});Json actual=Json::object();
            try{
                if(op=="buffered")actual["value"]=serialize(service.generate(request.dump(),key));
                else if(op=="pipeline")actual["value"]=serialize(service.generate_pipeline(request.dump(),key));
                else if(op=="begin")stream=service.stream(request.dump(),key);
                else if(op=="next"){stream->consumer_key(key);const auto value=stream->next();actual["value"]=value?serialize(*value):Json();}
                else if(op=="close"){stream->consumer_key(key);stream->close();}
                else if(op=="cancel"){stream->consumer_key(key);stream->request_cancel();stream->next();}
                else if(op=="mode")env.mode=action.at("value");
                else if(op=="warm")service.warm();
                else if(op=="drain"){while(!env.pending.empty()){auto work=std::move(env.pending.front());env.pending.erase(env.pending.begin());work();}}
                else if(op=="destroy")service.unload(key);
                else throw std::runtime_error("unknown service action");
            }catch(const TtsError &e){actual["error"]={{"category",e.category},{"message",e.what()}};}catch(const std::exception &e){actual["error"]={{"category","RuntimeError"},{"message",e.what()}};}
            actual["state"]=env.state(service);compare_json(actual,scenario.at("outcomes")[step],label+" action "+std::to_string(step));++step;++actions;
        }
        compare_json(env.events,scenario.at("events"),label+" events");
    return actions;
}
