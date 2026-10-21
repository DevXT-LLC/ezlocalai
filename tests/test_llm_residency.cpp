#include "ezlocalai/llm_residency.hpp"
#include "ezlocalai/llm_memory.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/llm_stream.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
using namespace ezlocalai;
namespace fs=std::filesystem;
using Json=nlohmann::json;
namespace {
void check(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
std::string read(const fs::path &p){std::ifstream f(p);return {std::istreambuf_iterator<char>(f),{}};}
void reference(const fs::path &fixture){
 const auto data=Json::parse(read(fixture));const auto root=fixture.parent_path();
 for(const auto &row:data.at("mode_cases")){
  LlmCatalog catalog;for(size_t i=0;i<row.at("estimates").size();++i){auto name="model"+std::to_string(i);catalog.available.push_back(name);catalog.configs[name].main_gpu=row.at("indices").at(i);}
  std::vector<std::string> calls;const bool result=select_llm_swap_residency(catalog,row.at("gpus").get<std::vector<double>>(),row.at("mode"),row.at("margin"),[&](const auto &name){calls.push_back(name);const auto &value=row.at("estimates").at(std::stoul(name.substr(5)));if(value.is_null())throw std::runtime_error("fixture estimate failure");return value.template get<double>();});
  check(result==(row.at("result")=="swap")&&Json(calls)==row.at("calls"),"residency oracle: "+row.dump());
 }
 for(const auto &row:data.at("memory_cases")){
  std::vector<std::string> projectors;for(const auto &p:row.at("projectors"))projectors.push_back((root/p.get<std::string>()).string());
  const auto path=(root/row.at("path").get<std::string>()).string();const auto cache=row.at("cache").get<std::string>();
  const auto actual=estimate_llm_memory_gb(path,row.at("context"),projectors,cache);const double expected=row.at("estimate");
  check(std::abs(actual-expected)<1e-10*std::max(1.,std::abs(expected)),"memory oracle actual="+std::to_string(actual)+" expected="+row.dump());
  check(hybrid_llm_kv_bytes_per_token(path,cache)==row.at("kv").get<int64_t>(),"hybrid KV oracle: "+row.dump());
 }
 std::cout<<data.at("mode_cases").size()<<" mode and "<<data.at("memory_cases").size()<<" memory cases passed\n";
}
void environment(){
 for(const auto &entry:environment_defaults)unsetenv(std::string(entry.first).c_str());
 for(auto key:{"EZLOCALAI_DOTENV_PATH","LLAMA_CPP_SERVER_URL","LLAMA_CPP_EMBEDDING_SERVER_URL","LLAMA_CPP_EMBEDDING_MODEL_PATH","LLAMA_CPP_MMPROJ_PATH","EZLOCALAI_TEST_CUDA_DEVICES","LLM_FIXTURE_FAIL_LAUNCH","LLM_FIXTURE_INFERENCE_RESULT"})unsetenv(key);
 for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_DOTENV_PATH",""},{"EZLOCALAI_API_KEY","none"},{"API_KEY",""},{"DEFAULT_MODEL","org/first,org/second,org/first"},{"N_PARALLEL","4"},{"LLM_SPECULATIVE_TYPE","none"},{"LLM_MODEL_RESIDENCY","swap"},{"LLAMA_CPP_CONTEXT","1024"},{"GPU_LAYERS","0"},{"LLAMA_CPP_START_TIMEOUT","2"},{"LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN},{"VOICE_SERVER",""},{"TEXT_SERVER",""},{"IMAGE_SERVER",""},{"EMBEDDING_SERVER",""},{"TTS_ENABLED","false"},{"STT_ENABLED","false"},{"EMBEDDING_ENABLED","false"},{"IMAGE_ENABLED","false"},{"VIDEO_ENABLED","false"},{"MUSIC_ENABLED","false"},{"LAZY_LOAD_VOICE","true"}})setenv(key.c_str(),value.c_str(),1);
}
std::shared_ptr<ResourceManager> manager(std::vector<double> capacities={}){
 auto p=native_resource_probes();p.gpu_count=[capacities]{return int(capacities.size());};p.total_per_gpu=p.free_per_gpu=[capacities]{return capacities;};p.free_vram=[](auto){return 24.;};p.allocator_stats=[]{return std::vector<CudaAllocatorStats>{};};p.system_ram=p.available_ram=[]{return 64.;};return std::make_shared<ResourceManager>(p);
}
int ports(){for(int trial=0;trial<100;++trial){std::vector<int> fds;int port=0;bool ok=true;for(int i=0;i<4;++i){int fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);fds.push_back(fd);sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);a.sin_port=htons(i?port+i:0);if(bind(fd,reinterpret_cast<sockaddr*>(&a),sizeof(a)))ok=false;if(!i){socklen_t n=sizeof(a);getsockname(fd,reinterpret_cast<sockaddr*>(&a),&n);port=ntohs(a.sin_port);}}for(auto fd:fds)close(fd);if(ok&&port<65532)return port;}throw std::runtime_error("ports unavailable");}
struct Fixture{
 fs::path directory;int port;
 Fixture():directory(fs::temp_directory_path()/("ezl-residency-"+std::to_string(getpid()))),port(ports()){
  fs::remove_all(directory);fs::create_directories(directory);auto path=directory/"model.gguf";std::ofstream(path)<<"GGUF fixture";setenv("LLAMA_CPP_MODEL_PATH",path.c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",directory.c_str(),1);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);
 }
 fs::path file(const std::string &prefix,int i=0){return directory/(prefix+"-"+std::to_string(port+i));}
 void wait_seen(int i){for(int n=0;n<1000&&!fs::exists(file("seen",i));++n)std::this_thread::sleep_for(std::chrono::milliseconds(5));check(fs::exists(file("seen",i)),"inference reached engine");}
 void release(int i){std::ofstream(file("release",i))<<"release";}
 int pid(int i){std::ifstream f(file("launch",i));int last=-1,p;while(f>>p)last=p;return last;}
 void reaped(){for(int i=0;i<3;++i){std::ifstream f(file("launch",i));int p;while(f>>p)check(kill(p,0)!=0,"managed process reaped");}}
 ~Fixture(){fs::remove_all(directory);}
};
Json lifecycle(LlamaCppInferenceBackend &backend){return Json::parse(backend.model_lifecycle_json());}
void managed(){
 environment();Fixture f;auto resources=manager();
 {
  LlamaCppInferenceBackend backend("org/first,org/second,org/first",resources);check(backend.swap_llm_residency(),"forced swap selected");std::string error;check(backend.warmup(&error),"warmup: "+error);
  auto state=lifecycle(backend);check(state["loaded_llms"]==Json::array({"org/first"})&&state["active_llm"]=="org/first","only first model warms");check(!fs::exists(f.file("launch",1))&&!fs::exists(f.file("launch",2)),"other processes not preloaded");
  const int first=f.pid(0);check(!backend.text_result(R"({"model":"org/second","prompt":"test"})","org/second",false).failure,"second model response");check(kill(first,0)!=0,"first process reaped before switch");check(lifecycle(backend)["loaded_llms"]==Json::array({"org/second"}),"only second resident");check(!resources->is_model_in_use(ModelType::Llm),"idle resource count");
  const int second=f.pid(1);auto stream=backend.lazy_text_result(R"({"model":"org/second","messages":[],"stream":true})","org/second",true);check(bool(stream.stream),"lazy stream created");bool blocked=false;try{backend.completion_for_replica(R"({"prompt":"test"})","org/first#2");}catch(const std::runtime_error &e){blocked=std::string(e.what()).find("Cannot switch")!=std::string::npos;}check(blocked&&kill(second,0)==0,"live stream prevents direct switch");stream.stream->close();
  check(!backend.text_result(R"({"prompt":"test"})","org/first#2",false).failure,"replica switch after close");check(kill(second,0)!=0&&lifecycle(backend)["loaded_llms"]==Json::array({"org/first"}),"only selected replica resident");
  auto restore=backend.unload_llms_for_media();check(lifecycle(backend)["loaded_llms"].empty(),"media handoff detaches selected model");restore(true);check(lifecycle(backend)["loaded_llms"]==Json::array({"org/first"})&&f.pid(2)>0,"media restore reloads selected replica only");
  auto events=lifecycle(backend)["recent_events"];int unloads=0;for(const auto &event:events)if(event["operation"]=="unload")++unloads;check(unloads>=3,"switch and handoff lifecycle recorded");
  auto pending=backend.lazy_text_result(R"({"messages":[{"role":"user","content":"block"}],"stream":true})","org/first#2",true);
  fs::remove(f.file("seen",2));
  auto next=std::async(std::launch::async,[&]{std::string chunk;return pending.stream->next(&chunk);});f.wait_seen(2);
  struct Release{Fixture &f;~Release(){f.release(2);}}release{f};
  pending.stream->close(); // The source cancellation join is bounded at five seconds.
  bool retained=false;try{backend.completion_for_replica(R"({"prompt":"test"})","org/second");}catch(const std::runtime_error &e){retained=std::string(e.what()).find("Cannot switch")!=std::string::npos;}
  check(retained&&kill(f.pid(2),0)==0,"cancelled running stream retains exclusive lease despite N_PARALLEL=4");
  f.release(2);pending.stream->abort_and_wait();next.get();check(!backend.text_result(R"({"prompt":"test"})","org/second",false).failure,"producer completion releases cancelled stream lease");
  std::ofstream(f.file("fail-launch",0))<<"fail";
  const auto failed=backend.text_result(R"({"prompt":"test"})","org/first",false);
  check(failed.body.find("error")!=std::string::npos&&lifecycle(backend)["loaded_llms"].empty()&&lifecycle(backend)["active_llm"].is_null(),"failed target load leaves truthful empty residency");
  check(!resources->is_model_loaded(ModelType::Llm),"failed switch clears old registry entry");
  fs::remove(f.file("fail-launch",0));check(!backend.text_result(R"({"prompt":"test"})","org/first",false).failure,"later request retries failed target");
 }
 f.reaped();
}
void scheduling(){
 environment();Fixture f;auto resources=manager();
 {
  ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=8;config.request_timeout_ms=10000;
  NativeServer server(config,std::make_unique<LlamaCppInferenceBackend>(config.default_model,resources),resources);
  const auto request=[&](const std::string &model,const std::string &prompt){return server.handle_request("POST","/v1/completions","",Json{{"model",model},{"prompt",prompt}}.dump());};
  auto first=std::async(std::launch::async,[&]{return request("org/first","block");});f.wait_seen(0);
  struct Release{Fixture &f;~Release(){for(int i=0;i<3;++i)f.release(i);}}release{f};
  auto second=std::async(std::launch::async,[&]{return request("org/second","test");});
  check(second.wait_for(std::chrono::milliseconds(100))==std::future_status::timeout&&!fs::exists(f.file("launch",1)),"other model queued without eviction");
  auto snapshot=Json::parse(server.handle_request("GET","/v1/resources","","").body);const auto slots=snapshot["slots"];check(slots["llm_queue_capacity"]==1&&snapshot["parallel"]["text_queue_capacity"]==1,"exclusive queue capacity");check(slots["cap_slots"]["text"]["capacity"]==1,"one global text slot");
  for(auto model:{"org/first","org/second"})check(slots["model_slots"][model]["capacity"]==1&&slots["model_slots"][model]["in_flight"]==1&&slots["model_slots"][model]["available"]==0,"all configured models advertise occupied slot");
  f.release(0);check(first.get().status==200&&second.get().status==200,"queued model finishes after first");
  snapshot=Json::parse(server.handle_request("GET","/v1/resources","","").body);check(snapshot["model_lifecycle"]["llm_model_residency"]=="swap"&&snapshot["model_lifecycle"]["loaded_llms"]==Json::array({"org/second"}),"resource lifecycle follows switch");
  const auto failed=request("org/first#2","fixture-error");check(failed.status==400&&failed.body==R"({"detail":{"type":"invalid_request_error","message":"fixture inference failed"}})","engine client error keeps its status without a fallback server: "+failed.body);check(request("org/second","test").status==200,"failed inference releases exclusive slot");
 }
 f.reaped();
}
void responsive_restore(bool embedding){
 environment();Fixture f;setenv("DEFAULT_MODEL","org/first",1);setenv("LLAMA_CPP_START_TIMEOUT","10",1);
 if(embedding){
  setenv("EMBEDDING_ENABLED","true",1);setenv("EMBEDDING_KEEP_LOADED","true",1);setenv("EMBEDDING_GPU_LAYERS","0",1);
  setenv("LLAMA_CPP_EMBEDDING_MODEL_PATH",(f.directory/"model.gguf").c_str(),1);
  setenv("LLAMA_CPP_EMBEDDING_SERVER_PORT",std::to_string(f.port+1).c_str(),1);
 }
 auto resources=manager();auto backend=std::make_unique<LlamaCppInferenceBackend>("org/first",resources);auto *pointer=backend.get();
 std::string error;check(backend->warmup(&error),"observer warmup: "+error);
 if(embedding)backend->warm_aux_model("embedding");
 ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::move(backend),resources);
 auto state=Json::parse(server.handle_request("GET","/v1/resources","","").body);
 check(state["model_lifecycle"]["loaded_llms"]==Json::array({"org/first"}),"initial observer contains resident model");
 auto restore=pointer->unload_llms_for_video();
 const auto hold=f.file("hold-startup");std::ofstream(hold)<<"hold";
 auto loading=std::async(std::launch::async,[&]{restore(true);});
 struct ReleaseStartup {fs::path path;~ReleaseStartup(){fs::remove(path);}} release{hold};
 for(int i=0;i<1000&&!fs::exists(f.file("waiting-startup"));++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));
 check(fs::exists(f.file("waiting-startup")),"restore reached controlled startup wait");
 std::future<bool> unloading;
 if(embedding){
  unloading=std::async(std::launch::async,[&]{return pointer->unload_aux_model("embedding");});
  // The destructor waits for the backend startup lock, after publishing the
  // empty pool. Status must remain responsive while both transitions wait.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
 }
 auto observing=std::async(std::launch::async,[&]{
  auto response=server.handle_request("GET","/v1/resources","","");check(response.status==200,"resources during restore");
  Json result=Json::parse(response.body);result["backend_contexts"]=pointer->loaded_model_contexts();
  return result;
 });
 const bool responsive=observing.wait_for(std::chrono::milliseconds(500))==std::future_status::ready;
 auto deciding=std::async(std::launch::async,[&]{return pointer->loaded_models_json();});
 const bool decision_waits=deciding.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout;
 // Always release the controlled child before joining either future, including
 // the deliberately failing pre-fix regression run.
 fs::remove(hold);loading.get();if(embedding)check(unloading.get(),"embedding unload completes");state=observing.get();
 const auto decision=Json::parse(deciding.get());
 check(decision_waits&&decision.contains("llm"),"admission decisions wait for current model state");
 check(responsive,"resource and router observations blocked by model startup");
 check(state["model_lifecycle"]["loaded_llms"].empty()&&state["model_lifecycle"]["active_llm"].is_null(),"restore snapshot must not advertise unloaded model");
 check(state["parallel"]["n_ctx"]==0&&state["backend_contexts"].empty(),"unloaded runtime context is not stale");
 if(embedding)check(state["model_pools"]["embedding"]["resident"]==0,"embedding removal is visible before destruction finishes");
 state=Json::parse(server.handle_request("GET","/v1/resources","","").body);
 check(state["model_lifecycle"]["loaded_llms"]==Json::array({"org/first"})&&state["parallel"]["n_ctx"].get<int>()>0,"observers refresh when restoration completes");
 check(!pointer->text_result(R"({"prompt":"test"})","org/first",false).failure,"restored model remains usable");
 std::cout<<"Resource and router metadata remain responsive during controlled LLM restoration\n";
}
void automatic(){
 environment();Fixture f;setenv("LLM_MODEL_RESIDENCY","auto",1);setenv("LLM_MODEL_RESIDENCY_MARGIN_GB","0",1);
 {LlamaCppInferenceBackend backend("org/first,org/second",manager({1}));check(backend.swap_llm_residency(),"auto estimates combined files above GPU capacity");check(lifecycle(backend)["vram_estimates_gb"].size()==2,"auto caches estimates");}
 {LlamaCppInferenceBackend backend("org/first,org/second",manager({24}));check(!backend.swap_llm_residency(),"auto keeps fitting models resident");}
 setenv("LLM_MODEL_RESIDENCY","swap",1);{LlamaCppInferenceBackend backend("org/first",manager());check(!backend.swap_llm_residency(),"single model forced resident");}
 setenv("LLM_MODEL_RESIDENCY","resident",1);{LlamaCppInferenceBackend backend("org/first,org/second",manager({1}));check(!backend.swap_llm_residency()&&lifecycle(backend)["vram_estimates_gb"].empty(),"forced resident skips estimation");std::string error;check(backend.warmup(&error)&&lifecycle(backend)["loaded_llms"].size()==2,"resident warms all models");}f.reaped();
}
}
int main(int argc,char **argv){try{if(argc>=4&&std::string(argv[1])=="--memory"){std::vector<std::string> projectors;for(int i=4;i<argc;++i)projectors.emplace_back(argv[i]);Json result;for(auto cache:{"fp16","q4_0","q8_0"})result[cache]=estimate_llm_memory_gb(argv[2],std::stoll(argv[3]),projectors,cache);std::cout<<result.dump()<<'\n';return 0;}check(argc==2,"fixture argument");reference(argv[1]);managed();scheduling();responsive_restore(false);responsive_restore(true);automatic();std::cout<<"Managed swap, stream guard, queue, media restoration, auto selection and resident controls passed\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
