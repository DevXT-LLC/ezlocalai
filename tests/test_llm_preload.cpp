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
void environment(){
 for(const auto &entry:environment_defaults)unsetenv(std::string(entry.first).c_str());
 for(auto key:{"EZLOCALAI_DOTENV_PATH","LLAMA_CPP_SERVER_URL","LLAMA_CPP_EMBEDDING_SERVER_URL","LLAMA_CPP_MMPROJ_PATH","EZLOCALAI_TEST_CUDA_DEVICES","LLM_FIXTURE_FAIL_LAUNCH","LLM_FIXTURE_INFERENCE_RESULT"})unsetenv(key);
 for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_DOTENV_PATH",""},{"EZLOCALAI_API_KEY","none"},{"API_KEY",""},{"DEFAULT_MODEL","org/first,org/second,org/first"},{"N_PARALLEL","4"},{"LLM_SPECULATIVE_TYPE","none"},{"LLM_MODEL_RESIDENCY","swap"},{"LLAMA_CPP_CONTEXT","1024"},{"GPU_LAYERS","0"},{"LLAMA_CPP_START_TIMEOUT","2"},{"LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN},{"VOICE_SERVER",""},{"TEXT_SERVER",""},{"IMAGE_SERVER",""},{"EMBEDDING_SERVER",""},{"TTS_ENABLED","false"},{"STT_ENABLED","false"},{"EMBEDDING_ENABLED","false"},{"IMAGE_ENABLED","false"},{"VIDEO_ENABLED","false"},{"MUSIC_ENABLED","false"},{"LAZY_LOAD_VOICE","true"}})setenv(key.c_str(),value.c_str(),1);
}
std::shared_ptr<ResourceManager> manager(std::vector<double> capacities={}){
 auto p=native_resource_probes();p.gpu_count=[capacities]{return int(capacities.size());};p.total_per_gpu=p.free_per_gpu=[capacities]{return capacities;};p.free_vram=[](auto){return 24.;};p.allocator_stats=[]{return std::vector<CudaAllocatorStats>{};};p.system_ram=p.available_ram=[]{return 64.;};return std::make_shared<ResourceManager>(p);
}
int ports(){for(int trial=0;trial<100;++trial){std::vector<int> fds;int port=0;bool ok=true;for(int i=0;i<4;++i){int fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);fds.push_back(fd);sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);a.sin_port=htons(i?port+i:0);if(bind(fd,reinterpret_cast<sockaddr*>(&a),sizeof(a)))ok=false;if(!i){socklen_t n=sizeof(a);getsockname(fd,reinterpret_cast<sockaddr*>(&a),&n);port=ntohs(a.sin_port);}}for(auto fd:fds)close(fd);if(ok&&port<65532)return port;}throw std::runtime_error("ports unavailable");}
struct Fixture{
 fs::path directory;int port;
 Fixture():directory(fs::temp_directory_path()/("ezl-preload-"+std::to_string(getpid()))),port(ports()){
  fs::remove_all(directory);fs::create_directories(directory);auto path=directory/"model.gguf";std::ofstream(path)<<"GGUF fixture";setenv("LLAMA_CPP_MODEL_PATH",path.c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",directory.c_str(),1);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);
 }
 fs::path file(const std::string &prefix,int i=0){return directory/(prefix+"-"+std::to_string(port+i));}
 void wait_seen(int i){for(int n=0;n<1000&&!fs::exists(file("seen",i));++n)std::this_thread::sleep_for(std::chrono::milliseconds(5));check(fs::exists(file("seen",i)),"inference reached engine");}
 void release(int i){std::ofstream(file("release",i))<<"release";}
 int pid(int i){std::ifstream f(file("launch",i));int last=-1,p;while(f>>p)last=p;return last;}
 void reaped(){for(int i=0;i<4;++i){std::ifstream f(file("launch",i));int p;while(f>>p)check(kill(p,0)!=0,"managed process reaped");}}
 ~Fixture(){fs::remove_all(directory);}
};
void reference(const fs::path &path) {
 const auto data=Json::parse(read(path));size_t count=0;
 for(const auto &row:data["cases"]){LlmCatalog catalog;catalog.available=row["models"].get<std::vector<std::string>>();catalog.automatic=row["automatic"].get<std::set<std::string>>();const auto failures=row["failures"].get<std::set<std::string>>();Json attempts=Json::array(),loaded=Json::array(),errors=Json::array();
  const auto failed=preload_llm_instances(catalog,row["swap"],[&](const auto &id){attempts.push_back(id);if(failures.count(id))throw std::runtime_error("fixture failure");loaded.push_back(id);},[&](const auto &id,const auto &message){check(message=="fixture failure","failure diagnostic");errors.push_back(id);});
  check(attempts==row["attempts"]&&loaded==row["loaded"]&&Json(failed)==row["failed"],"preload oracle case "+std::to_string(count++));check(errors.size()+loaded.size()==attempts.size(),"every load completed or reported failure");
 }
 std::cout<<count<<" Python preload cases passed\n";
}
Json snapshot(NativeServer &server){const auto response=server.handle_request("GET","/v1/resources","","");check(response.status==200,"resource snapshot");return Json::parse(response.body);}
void managed(bool original_failure,bool swap){
 environment();Fixture f;setenv("DEFAULT_MODEL","org/first-mtp,org/second-mtp",1);setenv("N_PARALLEL","2",1);setenv("LLM_SPECULATIVE_TYPE","mtp",1);setenv("LLM_MODEL_RESIDENCY",swap?"swap":"resident",1);
 std::ofstream(f.file("fail-launch",2))<<"fail";if(original_failure)std::ofstream(f.file("fail-launch",0))<<"fail";
 auto resources=manager();{
  auto backend=std::make_unique<LlamaCppInferenceBackend>("org/first-mtp,org/second-mtp",resources);auto *engine=backend.get();std::string error;check(engine->warmup(&error),"preload failures must not abort startup");
  const auto replicas=*engine->llm_replica_state();check(replicas.size()==4,"two generated replicas");check(!replicas[0].failed&&!replicas[1].failed&&!replicas[3].failed,"configured originals and later successful replica remain eligible");check(replicas[2].failed==!swap,"only failed preloaded automatic replica is excluded");
  if(!swap){check(fs::exists(f.file("probe",3)),"preload continues after earlier failure");check(error.find("first-mtp#2")!=std::string::npos,"failed replica diagnostic retained");}
  else check(!fs::exists(f.file("launch",1))&&!fs::exists(f.file("launch",2)),"swap leaves deferred instances untouched");
  ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=9;config.request_timeout_ms=10000;
  NativeServer server(config,std::move(backend),resources);auto state=snapshot(server);const int capacity=swap?1:3;
  check(state["parallel"]["text_queue_capacity"]==capacity&&state["slots"]["llm_queue_capacity"]==capacity,"queue uses surviving source capacity");
  check(state["slots"]["model_slots"]["org/first-mtp"]["capacity"]==1&&state["slots"]["model_slots"]["org/second-mtp"]["capacity"]==(swap?1:2),"router slots exclude failed replica");
  auto request=[&](const std::string &model,const std::string &prompt){return server.handle_request("POST","/v1/completions","",Json{{"model",model},{"prompt",prompt},{"disable_fallback",true}}.dump());};
  if(!swap){const auto failed=request("org/first-mtp#2","test");check(failed.status==500,"explicit failed replica rejected without hanging: "+failed.body);check(!fs::exists(f.file("seen",2)),"failed replica never selected");}
  if(original_failure){fs::remove(f.file("fail-launch",0));check(request("org/first-mtp","test").status==200,"configured original retries after preload failure");check(fs::exists(f.file("seen",0)),"retry reaches original engine");}
  if(!swap&&!original_failure){
   std::future<HttpResponse> first,second,third,fourth;
   struct Release{Fixture &f;~Release(){for(int i=0;i<4;++i)f.release(i);}}release{f};
   first=std::async(std::launch::async,[&]{return request("org/first-mtp","block");});f.wait_seen(0);
   second=std::async(std::launch::async,[&]{return request("org/second-mtp","block");});f.wait_seen(1);
   third=std::async(std::launch::async,[&]{return request("org/second-mtp#2","block");});f.wait_seen(3);
   fourth=std::async(std::launch::async,[&]{return request("org/first-mtp","test");});check(fourth.wait_for(std::chrono::milliseconds(100))==std::future_status::timeout,"fourth request waits behind surviving three slots");
   state=snapshot(server);check(state["slots"]["cap_slots"]["text"]["in_flight"]==3&&state["slots"]["cap_slots"]["text"]["queued"]==1,"three active and one queued advertised");
   for(int i=0;i<4;++i)f.release(i);check(first.get().status==200&&second.get().status==200&&third.get().status==200&&fourth.get().status==200,"surviving replicas and queue complete");
  }
  check(snapshot(server)["slots"]["cap_slots"]["text"]["in_flight"]==0,"all leases released");
 }f.reaped();
}
void configured_duplicate(){
 environment();Fixture f;setenv("DEFAULT_MODEL","org/first-mtp,org/first-mtp",1);setenv("N_PARALLEL","1",1);setenv("LLM_MODEL_RESIDENCY","resident",1);std::ofstream(f.file("fail-launch",1))<<"fail";auto resources=manager();{
 auto backend=std::make_unique<LlamaCppInferenceBackend>("org/first-mtp,org/first-mtp",resources);check(backend->warmup(),"failed explicit duplicate preload continues");const auto replica_state=*backend->llm_replica_state();for(const auto &r:replica_state)check(!r.failed,"explicit duplicate is not an automatic failure");
 ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::move(backend),resources);check(snapshot(server)["slots"]["model_slots"]["org/first-mtp"]["capacity"]==2,"explicit duplicate stays in capacity");fs::remove(f.file("fail-launch",1));const auto reply=server.handle_request("POST","/v1/completions","",R"({"model":"org/first-mtp#2","prompt":"test","disable_fallback":true})");check(reply.status==200&&fs::exists(f.file("seen",1)),"explicit configured duplicate retries");
 }f.reaped();
}
}
int main(int argc,char **argv){try{check(argc==2,"fixture required");reference(argv[1]);managed(false,false);managed(true,false);managed(true,true);configured_duplicate();std::cout<<"Failed automatic replicas, continued preload, queue/slots, original retries and swap deferral passed\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
