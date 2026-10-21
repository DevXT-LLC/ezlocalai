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
#include <condition_variable>
#include <cerrno>
#include <mutex>
#include <set>
#include <limits>
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
 Fixture():directory(fs::temp_directory_path()/("ezl-live-capacity-"+std::to_string(getpid()))),port(ports()){
  fs::remove_all(directory);fs::create_directories(directory);auto path=directory/"model.gguf";std::ofstream(path)<<"GGUF fixture";setenv("LLAMA_CPP_MODEL_PATH",path.c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",directory.c_str(),1);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);
 }
 fs::path file(const std::string &prefix,int i=0){return directory/(prefix+"-"+std::to_string(port+i));}
 void wait_seen(int i){for(int n=0;n<1000&&!fs::exists(file("seen",i));++n)std::this_thread::sleep_for(std::chrono::milliseconds(5));check(fs::exists(file("seen",i)),"inference reached engine");}
 void release(int i){std::ofstream(file("release",i))<<"release";}
 int pid(int i){std::ifstream f(file("launch",i));int last=-1,p;while(f>>p)last=p;return last;}
 void reaped(){for(int i=0;i<4;++i){std::ifstream f(file("launch",i));int p;while(f>>p)check(kill(p,0)!=0,"managed process reaped");}}
 ~Fixture(){fs::remove_all(directory);}
};
struct QueueControl {std::mutex mutex;std::condition_variable changed;std::set<int> released;bool all=false;};
void processor(const char *payload,char *result,size_t size,void *data){auto &c=*static_cast<QueueControl*>(data);const int id=std::stoi(payload);std::unique_lock<std::mutex> lock(c.mutex);c.changed.wait(lock,[&]{return c.all||c.released.count(id);});std::snprintf(result,size,"%d",id);}
struct QueueOwner {QueueControl control;ezl_queue *queue=ezl_queue_create(8,20);~QueueOwner(){{std::lock_guard<std::mutex> lock(control.mutex);control.all=true;control.changed.notify_all();}ezl_queue_destroy(queue);}};
void queue_reference(const fs::path &path){const auto fixture=Json::parse(read(path));QueueOwner owner;auto *queue=owner.queue;check(queue&&ezl_queue_set_concurrency(queue,1)==0&&ezl_queue_start(queue,processor,&owner.control)==0,"queue initialization and pre-start resize");check(ezl_queue_set_concurrency(queue,std::numeric_limits<size_t>::max())==ENOMEM,"unrepresentable worker allocation preserves live queue");int next=0;
 for(const auto &row:fixture["transitions"]){const auto action=row["action"].get<std::string>();const int value=row["value"];
  if(action=="enqueue")for(int i=0;i<value;++i){char id[64];check(ezl_queue_enqueue(queue,std::to_string(next++).c_str(),id,sizeof(id))==0,"enqueue controlled request");}
  else if(action=="resize")check(ezl_queue_set_concurrency(queue,value)==0,"resize live queue");
  else {std::lock_guard<std::mutex> lock(owner.control.mutex);owner.control.released.insert(value);owner.control.changed.notify_all();}
  const auto state=[&]{ezl_queue_snapshot s{};ezl_queue_get_snapshot(queue,&s);return Json{{"queue_size",s.queued},{"processing_count",s.processing},{"max_concurrent",s.max_concurrent_requests},{"total_processed",s.total_processed}};};
  for(int i=0;i<500&&state()!=row["snapshot"];++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));
  check(state()==row["snapshot"],"source queue transition "+action+" "+std::to_string(value)+" got "+state().dump());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));check(state()==row["snapshot"],"resized admission remains stable");
 }
 ezl_queue_stop(queue);check(ezl_queue_set_concurrency(queue,2)==0&&ezl_queue_start(queue,processor,&owner.control)==0,"stopped queue can resize and restart");
 std::cout<<fixture["transitions"].size()<<" Python queue transitions passed\n";
}
void loading_snapshot(){environment();Fixture f;setenv("LLM_FIXTURE_PROBE_BLOCK","1",1);setenv("N_PARALLEL","0",1);setenv("LLM_MAX_TOKENS","131072",1);setenv("LLAMA_CPP_CONTEXT","0",1);{
 LlamaCppInferenceBackend engine("org/first",manager());setenv("LLAMA_CPP_CONTEXT","32768",1);std::future<bool> warm;std::future<int64_t> snapshot;
 struct Release {Fixture &f;~Release(){std::ofstream(f.file("release-probe"))<<"release";}} release{f};
 warm=std::async(std::launch::async,[&]{return engine.warmup();});for(int i=0;i<1000&&!fs::exists(f.file("probe"));++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));check(fs::exists(f.file("probe")),"startup probe reached");
 snapshot=std::async(std::launch::async,[&]{return llm_replica_capacity(engine.llm_replica_state()->front());});check(snapshot.wait_for(std::chrono::milliseconds(100))==std::future_status::ready&&snapshot.get()==4,"capacity snapshot remains available during model load");std::ofstream(f.file("release-probe"))<<"release";check(warm.get(),"blocked startup finishes");check(llm_replica_capacity(engine.llm_replica_state()->front())==1,"successful startup publishes resolved slots");
 }unsetenv("LLM_FIXTURE_PROBE_BLOCK");f.reaped();}
void managed(){environment();Fixture f;setenv("DEFAULT_MODEL","org/first",1);setenv("N_PARALLEL","0",1);setenv("LLM_MAX_TOKENS","131072",1);setenv("LLAMA_CPP_CONTEXT","0",1);setenv("LLM_MODEL_RESIDENCY","resident",1);auto resources=manager();{
 auto backend=std::make_unique<LlamaCppInferenceBackend>("org/first",resources);auto *engine=backend.get();check(llm_replica_capacity(engine->llm_replica_state()->front())==4,"unloaded model uses configured auto capacity");setenv("LLAMA_CPP_CONTEXT","32768",1);check(engine->warmup(),"managed preload");check(llm_replica_capacity(engine->llm_replica_state()->front())==1,"loaded model exposes actual auto capacity");check(engine->loaded_model_contexts().at("org/first")==32768,"per-slot context uses actual parallel count");
 setenv("LLAMA_CPP_CONTEXT","0",1);ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::move(backend),resources);
 const auto state=[&]{auto r=server.handle_request("GET","/v1/resources","","");check(r.status==200,"resource response");return Json::parse(r.body);};
 auto current=state();check(current["parallel"]["n_parallel"]==1&&current["parallel"]["n_ctx"]==32768&&current["parallel"]["text_queue_capacity"]==1&&current["slots"]["llm_queue_capacity"]==1,"loaded runtime and startup queue");
 auto restore=engine->unload_llms_for_media();current=state();check(current["parallel"]["n_ctx"]==0&&current["parallel"]["n_parallel"]==1,"no active model reports empty context");check(current["parallel"]["text_queue_capacity"]==1&&current["slots"]["llm_queue_capacity"]==4,"resource response records old queue before growth");check(state()["parallel"]["text_queue_capacity"]==4,"resource query grows queue");
 restore(true);current=state();check(current["parallel"]["text_queue_capacity"]==4&&current["slots"]["llm_queue_capacity"]==1,"resource response records old queue before shrink");check(state()["parallel"]["text_queue_capacity"]==1,"restoration shrinks admission to actual slots");
 auto detached=engine->unload_llms_for_media();
 const auto reply=server.handle_request("POST","/v1/completions","",R"({"model":"org/first","prompt":"test","disable_fallback":true})");check(reply.status==200,"inference after detach");
 current=state();check(current["parallel"]["text_queue_capacity"]==4&&current["parallel"]["n_parallel"]==4&&current["parallel"]["n_ctx"]==131072,"enqueue and reload refresh configured auto capacity");check(current["parallel"]["per_slot_context"]==32768&&engine->loaded_model_contexts().at("org/first")==32768,"router per-slot contexts reflect live parallelism");detached(false);
 }f.reaped();
}
void resolved_stream_lease(){
 environment();Fixture f;setenv("DEFAULT_MODEL","org/first",1);setenv("LLM_MODEL_RESIDENCY","resident",1);setenv("LLAMA_CPP_EXTRA_ARGS","-np 1",1);auto resources=manager();{
  LlamaCppInferenceBackend backend("org/first",resources);check(backend.warmup()&&backend.llm_parallel_count()==1,"engine override resolves configured four slots to one");
  auto pending=backend.lazy_text_result(R"({"messages":[{"role":"user","content":"block"}],"stream":true})","org/first",true);
  auto next=std::async(std::launch::async,[&]{std::string chunk;return pending.stream->next(&chunk);});
  struct Release{Fixture &f;~Release(){f.release(0);}}release{f};f.wait_seen(0);
  pending.stream->close();
  check(resources->is_model_in_use(ModelType::Llm),"resolved single-slot stream retains engine lease after bounded cancellation join");
  f.release(0);pending.stream->abort_and_wait();next.get();
  check(!resources->is_model_in_use(ModelType::Llm),"producer completion releases resolved single-slot lease");
 }f.reaped();unsetenv("LLAMA_CPP_EXTRA_ARGS");
}
void stop_after_shrink(){QueueOwner owner;auto *q=owner.queue;check(ezl_queue_set_concurrency(q,3)==0&&ezl_queue_start(q,processor,&owner.control)==0,"start shrink/stop queue");for(int i=0;i<4;++i){char id[64];check(ezl_queue_enqueue(q,std::to_string(i).c_str(),id,sizeof(id))==0,"enqueue stop fixture");}ezl_queue_snapshot s{};for(int i=0;i<500;++i){ezl_queue_get_snapshot(q,&s);if(s.processing==3)break;std::this_thread::sleep_for(std::chrono::milliseconds(5));}check(s.processing==3&&ezl_queue_set_concurrency(q,1)==0,"shrink while three requests active");
 auto stopped=std::async(std::launch::async,[&]{ezl_queue_stop(q);});for(int i=0;i<500&&ezl_queue_is_running(q);++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));const int resize=ezl_queue_set_concurrency(q,2);
 {std::lock_guard<std::mutex> lock(owner.control.mutex);owner.control.all=true;owner.control.changed.notify_all();}check(stopped.wait_for(std::chrono::seconds(5))==std::future_status::ready,"shutdown drains resized queue");stopped.get();check(resize==ECANCELED,"resize during shutdown is nonblocking and rejected");ezl_queue_get_snapshot(q,&s);check(s.total_processed==4&&s.active==0,"shutdown leaves no stranded requests");
}
}
int main(int argc,char **argv){try{check(argc==2,"fixture required");queue_reference(argv[1]);stop_after_shrink();loading_snapshot();managed();resolved_stream_lease();std::cout<<"Live queue resize, runtime auto slots, unload/restore, response ordering, enqueue refresh and resolved stream lease passed\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
