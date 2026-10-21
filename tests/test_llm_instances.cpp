#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "fixture_ports.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <csignal>
#include <unistd.h>
using namespace ezlocalai;
using Json=nlohmann::json;
namespace fs=std::filesystem;
namespace {
void check(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
void env(const char *key,const std::string &value){setenv(key,value.c_str(),1);}
void environment(){
 for(const auto &[key,value]:environment_defaults)unsetenv(std::string(key).c_str());
 for(auto key:{"LLAMA_CPP_CONTEXT","LLAMA_CPP_EXTRA_ARGS","LLAMA_CPP_SERVER_URL","LLAMA_CPP_MMPROJ_PATH","LLAMA_CPP_MODEL_PATHS","LLM_FIXTURE_FAIL_LAUNCH","LLM_FIXTURE_PROBE_BLOCK"})unsetenv(key);
 env("EZLOCALAI_DOTENV_PATH","");env("EZLOCALAI_API_KEY","");env("ROUTER_URL","");env("CUDA_VISIBLE_DEVICES","");env("VOICE_SERVER","");env("TEXT_SERVER","");env("IMAGE_SERVER","");env("EMBEDDING_SERVER","");
 for(auto key:{"TTS_ENABLED","STT_ENABLED","EMBEDDING_ENABLED","IMAGE_ENABLED","VIDEO_ENABLED","MUSIC_ENABLED"})env(key,"false");
 env("LLM_SPECULATIVE_TYPE","none");env("N_PARALLEL","2");env("LLM_MAX_TOKENS","65536");env("LAZY_LOAD_VOICE","true");
}
std::shared_ptr<ResourceManager> manager(){auto p=native_resource_probes();p.gpu_count=[]{return 0;};p.total_per_gpu=p.free_per_gpu=[]{return std::vector<double>{};};p.free_vram=[](auto){return 24.;};p.allocator_stats=[]{return std::vector<CudaAllocatorStats>{};};p.system_ram=p.available_ram=[]{return 64.;};return std::make_shared<ResourceManager>(p);}
Json slots(NativeServer &server){auto response=server.handle_request("GET","/v1/resources","","");check(response.status==200,"resource status");return Json::parse(response.body).at("slots");}
struct Backend:InferenceBackend{
 std::vector<LlmReplica> replicas;bool swap=false;
 std::string chat(const std::string &)override{return "{}";}
 std::string completion(const std::string &)override{return "{}";}
 std::string embedding(const std::string &)override{return "{}";}
 std::optional<std::vector<LlmReplica>> llm_replica_state()const override{return replicas;}
 bool swap_llm_residency()const override{return swap;}
};
void reference(const Json &fixture){
 size_t index=0;
 for(const auto &row:fixture.at("cases")){
  environment();std::string models;for(const auto &model:row.at("models")){if(!models.empty())models+=',';models+=model.get<std::string>();}env("DEFAULT_MODEL",models);
  auto backend=std::make_unique<Backend>();backend->swap=row.at("swap");
  for(const auto &r:row.at("replicas"))backend->replicas.push_back({r.at("id"),r.at("source"),r.at("parallel"),r.at("context"),r.at("actual_parallel").is_null()?std::nullopt:std::optional<int64_t>(r.at("actual_parallel")),false,r.at("failed"),r.at("resident")});
  ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::move(backend),manager());const auto actual=slots(server);
  check(actual==row.at("slots"),"source snapshot "+std::to_string(index)+": "+actual.dump()+" expected "+row.at("slots").dump());++index;
 }
 std::cout<<index<<" complete Python slot snapshots passed\n";
}
void managed(bool automatic,bool swap){
 environment();const auto path=fs::path("engines")/(std::string(automatic?"auto":"explicit")+(swap?"-swap":"-resident"));fs::create_directories(path);
 const int port=fixture_ports(4);std::ofstream(path/"model.gguf")<<"fixture";
 const std::string models=automatic?"org/model-mtp":"org/model,org/model";const std::string model=automatic?"org/model-mtp":"org/model";
 env("DEFAULT_MODEL",models);env("N_PARALLEL",automatic?"3":"1");env("LLM_SPECULATIVE_TYPE",automatic?"mtp":"none");env("LLM_MODEL_RESIDENCY",swap?"swap":"resident");env("LLAMA_CPP_CONTEXT","4096");env("GPU_LAYERS","0");env("LLAMA_CPP_MODEL_PATH",fs::absolute(path/"model.gguf").string());env("LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN);env("LLAMA_CPP_START_TIMEOUT","2");env("LLAMA_CPP_SERVER_PORT",std::to_string(port));env("EMBEDDING_FIXTURE_DIR",fs::absolute(path).string());
 if(automatic)std::ofstream(path/("fail-launch-"+std::to_string(port+1)))<<"fail";
 {auto resources=manager();auto backend=std::make_unique<LlamaCppInferenceBackend>(models,resources);auto *engine=backend.get();check(engine->warmup(),"managed warmup");
  ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::move(backend),resources);
  const auto verify=[&](bool resident){const auto current=slots(server);for(const auto &slot:{current.at("model_slots").at(model),current.at("cap_slots").at("text")}){
   if(resident&&!swap)check(slot.at("instances")==2,"two successful managed instances reported");else check(!slot.contains("instances"),"unloaded or swap instances omitted");
  }};
  verify(true);auto restore=engine->unload_llms_for_media();verify(false);restore(true);verify(true);
 }
 for(int i=0;i<4;++i){std::ifstream input(path/("launch-"+std::to_string(port+i)));int pid;while(input>>pid)check(kill(pid,0)!=0,"managed instance process reaped");}
}
}
int main(int argc,char **argv){const auto old=fs::current_path();fs::path temporary;try{
 check(argc==2,"fixture required");std::ifstream input(argv[1]);Json fixture;input>>fixture;
 temporary=fs::temp_directory_path()/("ezlocalai-instances-"+std::to_string(getpid()));fs::create_directories(temporary);fs::current_path(temporary);
 for(auto name:{"vision","vision2"}){fs::create_directories(fs::path("models")/name);std::ofstream(fs::path("models")/name/"mmproj.gguf")<<"fixture";}
 reference(fixture);managed(false,false);managed(true,false);managed(true,true);
 fs::current_path(old);fs::remove_all(temporary);std::cout<<"Resident instance metadata, failed preload, explicit duplicates, unload/restore and swap checks passed\n";return 0;
 }catch(const std::exception &error){fs::current_path(old);std::cerr<<error.what()<<"\nArtifacts: "<<temporary<<'\n';return 1;}}
