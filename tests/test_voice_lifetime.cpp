#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "whisper_transcription_test.hpp"
#include <set>
using namespace transcription_test;
struct Environment {
    Json events=Json::array();std::string failure;
    std::map<std::string,std::map<const void *,int>> ids;
    std::map<std::string,std::map<int,int>> closed;
    std::vector<SttPoolHandle> stt_models;std::vector<TtsPoolHandle> tts_models;
    std::vector<std::function<void()>> pending;
    void event(const std::string &name,Json fields=Json::object()){fields["event"]=name;events.push_back(fields);}
    int loaded(const std::string &kind,const void *model){const int id=closed[kind].size()+1;ids[kind][model]=id;closed[kind][id]=0;event("load",{{"kind",kind},{"id",id}});return id;}
    void infer(const std::string &kind,int id){event("infer",{{"kind",kind},{"id",id}});if(failure==kind)throw TtsError("RuntimeError","fixture inference failure");}
    SttPoolOperations stt(std::shared_ptr<ResourceManager> resources){
        auto ops=stt_pool_resources({},resources);ops.schedule_cleanup=[&](auto work){pending.push_back(std::move(work));};
        ops.load=[&](const auto &){auto model=std::make_shared<SttPoolModel>();const auto id=loaded("stt",model.get());stt_models.push_back(model);model->device=[]{return "cpu";};model->close=[&,id]{++closed["stt"][id];event("close",{{"kind","stt"},{"id",id}});};model->transcribe_bytes=[&,id](const auto &,const auto &){infer("stt",id);return Json("transcript").dump();};return model;};return ops;
    }
    TtsPoolOperations tts(std::shared_ptr<ResourceManager> resources){
        auto ops=tts_pool_resources({},resources);ops.can_load=[]{return ModelAdmission{true,"cpu","fixture"};};ops.schedule_cleanup=[&](auto work){pending.push_back(std::move(work));};
        ops.load=[&](const auto &){auto model=std::make_shared<TtsPoolModel>();const auto id=loaded("tts",model.get());tts_models.push_back(model);model->device=[]{return "cpu";};model->closed=[&,id]{return closed["tts"][id]!=0;};model->close=[&,id]{++closed["tts"][id];event("close",{{"kind","tts"},{"id",id}});};model->generate=[&,id](const auto &){infer("tts",id);return "YWJj";};return model;};return ops;
    }
    template<class Snapshot> Json pool(const std::string &kind,const Snapshot &pool){
        auto id=[&](const auto &model){return model?Json(ids[kind].at(model.get())):Json();};
        Json out={{"primary",id(pool.primary)},{"resident",Json::array()},{"transient",Json::array()},{"available",Json::array()},{"leases",Json::array()},{"closed",Json::object()}};
        for(const auto &m:pool.resident)out["resident"].push_back(id(m));for(const auto &m:pool.transient)out["transient"].push_back(id(m));for(const auto &m:pool.available)out["available"].push_back(id(m));for(const auto &m:pool.leases)out["leases"].push_back(id(m.second));for(const auto &[key,value]:closed[kind])out["closed"][std::to_string(key)]=value;return out;
    }
    Json state(const NativeServer &server,const ResourceManager &resources){
        Json out={{"pending",pending.size()},{"pools",{{"stt",pool("stt",*server.stt_pool_snapshot())},{"tts",pool("tts",*server.tts_pool_snapshot())}}},{"resources",Json::object()}};
        for(const auto &r:resources.loaded_models())out["resources"][model_type_name(r.type)]={{"name",r.name},{"device",r.device},{"vram",r.vram_gb},{"active",r.active_requests}};return out;
    }
};
struct Backend:InferenceBackend {
    EchoInferenceBackend echo{"fixture-text"};
    Environment &env;explicit Backend(Environment &env):env(env){}
    std::string chat(const std::string &body)override{return echo.chat(body);}
    std::string completion(const std::string &body)override{return echo.completion(body);}
    std::string embedding(const std::string &body)override{return echo.embedding(body);}
    std::function<void(bool)> unload_llms_for_media()override{env.event("unload");return [this](bool){env.event("restore");};}
};
int main(int argc,char **argv){try{
    expect(argc==3,"source and voice lifetime fixture");gzFile file=gzopen(argv[2],"rb");expect(file,"open lifetime fixture");std::string raw;char buffer[65536];int n;while((n=gzread(file,buffer,sizeof(buffer)))>0)raw.append(buffer,n);expect(n==0&&gzclose(file)==Z_OK,"complete lifetime fixture");std::vector<Json> cases;std::istringstream input(raw);for(std::string line;std::getline(input,line);)if(!line.empty())cases.push_back(Json::parse(line));
    for(const auto &[file,key]:std::map<std::string,std::string>{{"Pipes.py","source_sha256"},{"app.py","app_sha256"},{"ModelSettings.py","settings_sha256"},{"ezlocalai/LlamaTTS.py","llama_sha256"}})expect(hash(read(fs::path(argv[1])/file))==cases[0].at(key),"unchanged lifetime source "+file);
    for(const auto &[file,sha]:cases[0].at("extra_sources").items())expect(hash(read(fs::path(argv[1])/file))==sha,"unchanged lifetime source "+file);
    expect(cases.size()-1==cases[0].at("cases"),"all lifetime cases");Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);size_t actions=0,failed=0;
    std::set<std::string> scenario_environment;
    for(size_t i=1;i<cases.size();++i){
        for(const auto &[key,value]:cases[i].at("config").at("environment").items())scenario_environment.insert(key);
        for(const auto &action:cases[i].at("actions"))if(action.at("op")=="env")
            for(const auto &[key,value]:action.at("values").items())scenario_environment.insert(key);
    }
    for(size_t i=1;i<cases.size();++i)try{const auto &row=cases[i],&settings=row.at("config");Environment env;
        for(const auto &key:scenario_environment)unsetenv(key.c_str());
        for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
        for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_API_KEY",""},{"WHISPER_MODEL","fixture-model"},{"QWEN_TTS_MODEL","fixture-tts"},{"VOICE_SERVER",""},{"VOICE_SERVER_API_KEY","original-key"},{"LAZY_LOAD_VOICE","true"},{"ROUTER_URL",""},{"FALLBACK_SERVER",""},{"TTS_ENABLED","true"},{"STT_ENABLED","true"},{"QWEN_TTS_ALLOW_CPU_FALLBACK","true"},{"EZLOCALAI_URL",""}})setenv(key.c_str(),value.c_str(),1);
        auto configure=[](const Json &values){for(const auto &[key,value]:values.items())setenv(key.c_str(),value.get<std::string>().c_str(),1);};configure(settings.at("environment"));std::string models;for(const auto &model:settings.at("models")){if(!models.empty())models+=',';models+=model.get<std::string>();}setenv("DEFAULT_MODEL",models.empty()?"none":models.c_str(),1);for(const auto *key:{"STT_N_PARALLEL","TTS_N_PARALLEL"})setenv(key,std::to_string(settings.at("capacity").get<int>()).c_str(),1);
        auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 16.;};probes.now=[]{return 100.;};auto resources=std::make_shared<ResourceManager>(probes);ezl_env_config config{};ezl_load_config(&config);
        NativeServer server(config,std::make_unique<Backend>(env),resources,[&](auto r){return env.stt(r);},[&](auto r){return env.tts(r);},{},settings.at("precache").get<bool>());
        auto check=[&](size_t at){const auto label="voice lifetime "+std::to_string(i)+" action "+std::to_string(at);compare_json(env.state(server,*resources),row.at("outcomes")[at].at("state"),label+" state");compare_json(env.events,row.at("outcomes")[at].at("events"),label+" events");++actions;};check(0);
        for(size_t j=0;j<row.at("actions").size();++j){const auto &action=row.at("actions")[j];const auto op=action.at("op");
            if(op=="env")configure(action.at("values"));else if(op=="drain"){while(!env.pending.empty()){auto work=std::move(env.pending.front());env.pending.erase(env.pending.begin());work();}}
            else if(op=="request"){env.failure=action.value("failure",std::string());const auto result=server.handle_request("POST","/v1/audio/speech","",R"({"input":"data:audio/wav;base64,YWJj"})","application/json");const auto &expected=row.at("outcomes")[j+1].at("response");const auto label="voice lifetime "+std::to_string(i)+" response "+std::to_string(j);compare_json(result.status,expected.at("status"),label+" status");compare_json(result.content_type,expected.at("mime"),label+" MIME");if(result.content_type=="application/json")compare_json(Json::parse(result.body),Json::parse(unhex(expected.at("hex"))),label+" body");else compare_json(result.body,unhex(expected.at("hex")),label+" body");}
            check(j+1);
        }
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';++failed;}
    expect(failed==0,std::to_string(failed)+" lifetime scenarios failed");std::cout<<cases.size()-1<<" source startup/configuration scenarios and "<<actions<<" state/event comparisons match\n";return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
