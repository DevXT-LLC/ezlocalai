#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <unistd.h>
namespace {
using namespace ezlocalai;using Json=nlohmann::json;namespace fs=std::filesystem;
void check(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
class Backend:public InferenceBackend {
public:
    std::string chat(const std::string &)override{return "{}";}
    std::string completion(const std::string &)override{return "{}";}
    std::string embedding(const std::string &)override{return "{}";}
    std::string loaded_models_json()const override{return R"({"img":{"name":"approximation"}})";}
};
struct Gate {std::mutex mutex;std::condition_variable changed;bool entered=false,open=false;
    void wait(){std::unique_lock<std::mutex> lock(mutex);entered=true;changed.notify_all();changed.wait(lock,[&]{return open;});}
    void ready(){std::unique_lock<std::mutex> lock(mutex);check(changed.wait_for(lock,std::chrono::seconds(3),[&]{return entered;}),"speech entered synthesis");}
    void release(){std::lock_guard<std::mutex> lock(mutex);open=true;changed.notify_all();}
};
}
int main(int argc,char **argv){try{
    check(argc==2,"source cases required");std::ifstream input(argv[1]);Json fixture;input>>fixture;
    char pattern[]="/tmp/ezlocalai-speech-resources-XXXXXX";check(mkdtemp(pattern),"temporary directory");const auto previous=fs::current_path();fs::current_path(pattern);
    struct Cleanup{fs::path previous,path;~Cleanup(){fs::current_path(previous);std::error_code ec;fs::remove_all(path,ec);}}cleanup{previous,pattern};
    for(const auto &entry:environment_defaults)unsetenv(std::string(entry.first).c_str());
    for(const auto &[name,value]:std::map<std::string,std::string>{{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","none"},{"VOICE_SERVER",""},{"FALLBACK_SERVER",""},{"TTS_ENABLED","true"},{"STT_ENABLED","true"},{"LAZY_LOAD_VOICE","false"},{"TTS_UNLOAD_LLM_DURING_GENERATION","false"},{"STT_UNLOAD_LLM_DURING_GENERATION","false"},{"TTS_N_PARALLEL","2"},{"STT_N_PARALLEL","2"},{"EMBEDDING_N_PARALLEL","3"},{"WHISPER_MODEL","large-v3-turbo"}})setenv(name.c_str(),value.c_str(),1);
    auto probes=native_resource_probes();probes.gpu_count=[] {return 0;};probes.total_per_gpu=probes.free_per_gpu=[] {return std::vector<double>{};};probes.free_vram=[](auto){return 0.;};probes.allocator_stats=[] {return std::vector<CudaAllocatorStats>{};};probes.system_ram=probes.available_ram=[] {return 16.;};probes.now=[] {return 100.;};
    auto resources=std::make_shared<ResourceManager>(probes);
    Gate gate;int tts_count=0,stt_count=0;
    TtsPoolFactory tts=[&](auto manager){auto ops=tts_pool_resources({},manager);ops.can_load=[] {return ModelAdmission{true,"cpu","fixture"};};ops.schedule_cleanup=[](auto work){work();};
        ops.load=[&](const auto &){const auto device=tts_count++==0?"cuda:0":"cpu";auto model=std::make_shared<TtsPoolModel>();model->device=[device]{return device;};model->closed=[] {return false;};model->close=[] {};model->generate=[&](const auto &){gate.wait();return "UklGRg==";};return model;};return ops;};
    SttPoolFactory stt=[&](auto manager){auto ops=stt_pool_resources({},manager);ops.can_load=[] {};ops.schedule_cleanup=[](auto work){work();};ops.load=[&](const auto &){const auto device=stt_count++==0?"cuda:1":"cpu";auto model=std::make_shared<SttPoolModel>();model->device=[device]{return device;};model->close=[] {};return model;};return ops;};
    ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::make_unique<Backend>(),resources,stt,tts,{},true);
    resources->register_model(ModelType::Image,"registered image","cpu",0);
    auto compare=[&](size_t index){const auto response=server.handle_request("GET","/v1/resources","","");check(response.status==200,"resources response");
        auto actual=Json::parse(response.body);check(response.body.find("approximation")==std::string::npos,"registered resource replaces backend duplicate: "+response.body);check(actual["loaded_models"]["img"]["name"]=="registered image","all registered modalities reported");actual["loaded_models"].erase("img");
        const auto &row=fixture.at("cases").at(index);check(actual.at("loaded_models")==row.at("loaded_models"),row.at("name").get<std::string>()+" resources: "+actual.at("loaded_models").dump());
        check(actual.at("model_pools")==row.at("model_pools"),row.at("name").get<std::string>()+" pools: "+actual.at("model_pools").dump());
    };
    compare(0);setenv("EMBEDDING_N_PARALLEL","9",1); // Pool configuration is captured at construction.
    auto request=[&]{return server.handle_request("POST","/v1/audio/speech","",R"({"input":"hello"})","application/json");};
    auto pending=std::async(std::launch::async,request);struct Release{Gate &gate;~Release(){gate.release();}}release{gate};
    gate.ready();compare(1);gate.release();check(pending.get().status==200,"buffered request completes");compare(2);
    check(request().status==200,"rotate resident queue before unloading");
    setenv("LAZY_LOAD_VOICE","true",1);check(request().status==200,"first resident unloads");compare(3);check(request().status==200,"second resident unloads");compare(4);
    std::cout<<fixture.at("cases").size()<<" source speech resource/pool snapshots match across preload, activity and unload\n";return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
