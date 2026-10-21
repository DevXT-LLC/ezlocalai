#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include "fixture_ports.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace ezlocalai;
using Json=nlohmann::json;
namespace fs=std::filesystem;
namespace {
void check(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
void env(const char *name,const std::string &value){setenv(name,value.c_str(),1);}
std::shared_ptr<ResourceManager> manager(){auto p=native_resource_probes();p.gpu_count=[]{return 0;};p.total_per_gpu=p.free_per_gpu=[]{return std::vector<double>{};};p.free_vram=[](auto){return 0.;};p.allocator_stats=[]{return std::vector<CudaAllocatorStats>{};};p.system_ram=p.available_ram=[]{return 64.;};return std::make_shared<ResourceManager>(p);}
void environment(const Json &row){
 for(const auto &[key,value]:environment_defaults)unsetenv(std::string(key).c_str());
 for(auto key:{"EZLOCALAI_PRELOAD_MODEL_ON_START","LLAMA_CPP_SERVER_URL","LLAMA_CPP_CONTEXT","LLAMA_CPP_EXTRA_ARGS","LLAMA_CPP_MMPROJ_PATH","LLAMA_CPP_MODEL_PATHS"})unsetenv(key);
 env("EZLOCALAI_DOTENV_PATH","");env("EZLOCALAI_API_KEY","");env("ROUTER_URL","");env("CUDA_VISIBLE_DEVICES","");env("IMAGE_ENABLED","false");env("VIDEO_ENABLED","false");env("MUSIC_ENABLED","false");env("FALLBACK_SERVER","");env("EZLOCALAI_NATIVE_BACKEND","echo");
 for(const auto &[key,value]:row.at("environment").items())env(key.c_str(),value.get<std::string>());
 std::string models;for(const auto &model:row.at("models")){if(!models.empty())models+=',';models+=model.get<std::string>();}env("DEFAULT_MODEL",models.empty()?"none":models);
}
struct State{
 Json events=Json::array();std::map<std::string,int> counts;std::string failure;int failure_slot;
 int load(const std::string &kind){const int slot=++counts[kind];events.push_back(kind+":load:"+std::to_string(slot));if(failure==kind&&failure_slot==slot)throw std::runtime_error("fixture load failure");return slot;}
 TtsPoolOperations tts(std::shared_ptr<ResourceManager> resources){auto ops=tts_pool_resources({},resources);ops.can_load=[]{return ModelAdmission{true,"cpu","fixture"};};ops.schedule_cleanup=[](auto work){work();};ops.load=[this](const auto &){load("tts");auto model=std::make_shared<TtsPoolModel>();model->device=[]{return "cpu";};return model;};return ops;}
 SttPoolOperations stt(std::shared_ptr<ResourceManager> resources){auto ops=stt_pool_resources({},resources);ops.can_load=[]{return ModelAdmission{true,"cpu","fixture"};};ops.schedule_cleanup=[](auto work){work();};ops.load=[this](const auto &){load("stt");auto model=std::make_shared<SttPoolModel>();model->device=[]{return "cpu";};return model;};return ops;}
};
struct Backend:InferenceBackend{
 State &state;size_t models;EmbeddingPool embeddings;
 Backend(State &s,size_t n):state(s),models(n),embeddings(embedding_parallel_slots(),[&s](size_t){return s.load("embedding");},[](auto){}){}
 std::string chat(const std::string &)override{return "{}";}
 std::string completion(const std::string &)override{return "{}";}
 std::string embedding(const std::string &)override{return "{}";}
 bool warmup(std::string *)override{for(size_t i=0;i<models;++i)state.load("llm");return true;}
 void warm_aux_model(const std::string &kind)override{check(kind=="embedding","startup auxiliary kind");embeddings.warm();}
};
void scenario(const Json &row,const std::optional<std::string> &preload={}){
 environment(row);if(preload)env("EZLOCALAI_PRELOAD_MODEL_ON_START",*preload);
 State state;state.failure=row.at("failure");state.failure_slot=row.at("failure_slot");auto backend=std::make_unique<Backend>(state,row.at("models").size());auto *model=backend.get();
 ezl_env_config config{};ezl_load_config(&config);std::unique_ptr<NativeServer> server;Json error;
 try{server=create_native_server(config,std::move(backend),manager(),[&](auto r){return state.stt(r);},[&](auto r){return state.tts(r);},row.at("precache").get<bool>());}catch(const std::runtime_error &){error="RuntimeError";}
 Json expected=row.at("events");if(preload&&(*preload=="false"||preload->empty())){Json filtered=Json::array();for(const auto &event:expected)if(event.get<std::string>().rfind("llm:",0)!=0)filtered.push_back(event);expected=filtered;}
 const std::string label=row.at("name");check(state.events==expected,label+": events="+state.events.dump()+" expected="+expected.dump());check(error==row.at("error"),label+": startup error");
 if(server){const Json residents={{"tts",server->tts_pool_snapshot()->resident.size()},{"stt",server->stt_pool_snapshot()->resident.size()},{"embedding",model->embeddings.snapshot().resident.size()}};check(residents==row.at("resident"),label+": resident pools "+residents.dump());}
}
struct Child{
 pid_t pid=-1;
 ~Child(){if(pid>0){kill(pid,SIGTERM);int status;for(int n=0;n<200;++n){if(waitpid(pid,&status,WNOHANG)==pid){pid=-1;return;}std::this_thread::sleep_for(std::chrono::milliseconds(10));}kill(pid,SIGKILL);waitpid(pid,&status,0);}}
};
void executable(const Json &base,const fs::path &binary){
 for(const auto &mode:{"default","voice","voice-text","forwarded","explicit-lazy"}){
  environment(base);env("EZLOCALAI_NATIVE_BACKEND","llama.cpp");for(auto key:{"TTS_ENABLED","STT_ENABLED","EMBEDDING_ENABLED"})env(key,"false");
  env("VOICE_SERVER",std::string(mode).find("voice")==0?"true":"");env("TEXT_SERVER",std::string(mode)=="voice-text"?"true":std::string(mode)=="forwarded"?"http://127.0.0.1:1":"");
  if(std::string(mode)=="explicit-lazy")env("EZLOCALAI_PRELOAD_MODEL_ON_START","false");
  const auto dir=fs::absolute(mode);fs::create_directories(dir);std::ofstream(dir/"model.gguf")<<"fixture";const int engine_port=fixture_ports(2),port=fixture_ports();
  env("LLAMA_CPP_MODEL_PATH",(dir/"model.gguf").string());env("LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN);env("LLAMA_CPP_SERVER_PORT",std::to_string(engine_port));env("LLAMA_CPP_START_TIMEOUT","2");env("GPU_LAYERS","0");env("EMBEDDING_FIXTURE_DIR",dir.string());env("PORT",std::to_string(port));env("EZLOCALAI_PORT",std::to_string(port));env("HOST","127.0.0.1");
  {Child child;child.pid=fork();check(child.pid>=0,"fork worker");if(!child.pid){const int fd=open((dir/"worker.log").c_str(),O_WRONLY|O_CREAT|O_TRUNC,0600);if(fd<0)_exit(126);dup2(fd,1);dup2(fd,2);close(fd);execl(binary.c_str(),binary.c_str(),nullptr);_exit(127);}
   bool ready=false;for(int i=0;i<100;++i){if(run_process({"curl","-fsS","--max-time","1","http://127.0.0.1:"+std::to_string(port)+"/health"},true,true).status==0){ready=true;break;}std::this_thread::sleep_for(std::chrono::milliseconds(50));}
   check(ready,"worker ready in mode "+std::string(mode));const bool expected=std::string(mode)=="default"||std::string(mode)=="voice-text";check(fs::exists(dir/("probe-"+std::to_string(engine_port)))==expected,"executable preload mode "+std::string(mode));
  }
  std::ifstream launches(dir/("launch-"+std::to_string(engine_port)));int pid;while(launches>>pid)check(kill(pid,0)!=0,"worker shutdown reaps preload process");
 }
}
}
int main(int argc,char **argv){const auto old=fs::current_path();fs::path directory;try{check(argc==3,"startup fixture and worker executable required");std::ifstream input(argv[1]);Json fixture;input>>fixture;const auto binary=fs::absolute(argv[2]);directory=fs::temp_directory_path()/("ezlocalai-startup-"+std::to_string(getpid()));fs::create_directories(directory);fs::current_path(directory);
 for(const auto &row:fixture.at("cases"))scenario(row);for(const auto &value:{"false","","true"})scenario(fixture.at("cases").front(),std::string(value));executable(fixture.at("cases").front(),binary);
 fs::current_path(old);fs::remove_all(directory);std::cout<<fixture.at("cases").size()<<" source startup traces, native lazy opt-out and five executable preload modes passed\n";return 0;}catch(const std::exception &e){fs::current_path(old);std::cerr<<e.what()<<"\nArtifacts: "<<directory<<'\n';return 1;}}
