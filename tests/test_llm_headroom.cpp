#include "ezlocalai/llm_residency.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "fixture_ports.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
using namespace ezlocalai;using Json=nlohmann::json;namespace fs=std::filesystem;
namespace {
void check(bool yes,const std::string &message){if(!yes)throw std::runtime_error(message);}
double number(const Json &value){return value.is_string()?std::stod(value.get<std::string>()):value.get<double>();}
bool same(double a,double b){return (std::isnan(a)&&std::isnan(b))||a==b||std::abs(a-b)<1e-10;}
ModelType type(const std::string &name){for(auto t:{ModelType::Llm,ModelType::VisionLlm,ModelType::Tts,ModelType::Stt,ModelType::Image,ModelType::Video,ModelType::Music,ModelType::Embedding})if(name==model_type_name(t))return t;throw std::runtime_error("model type");}
ResourceProbes probes(){auto p=native_resource_probes();p.gpu_count=[]{return 0;};p.total_per_gpu=p.free_per_gpu=[]{return std::vector<double>{};};p.free_vram=[](auto){return 0.;};p.allocator_stats=[]{return std::vector<CudaAllocatorStats>{};};p.system_ram=p.available_ram=[]{return 64.;};return p;}
void reference(const fs::path &path){std::ifstream input(path);Json data;input>>data;
 for(const auto &row:data.at("headroom")){
  std::vector<std::string> events;const auto actual=llm_load_headroom(row.at("main_gpu"),{
   [&]{events.push_back("estimate");if(row.at("estimate").is_null())throw std::runtime_error("estimate failed");return number(row.at("estimate"));},
   [&]{events.push_back("per_gpu");return row.at("free").get<std::vector<double>>();},[&]{events.push_back("total");return number(row.at("total"));},[&]{events.push_back("margin");return row.at("margin").get<std::string>();}});
  if(actual.reclaim_gb>0)events.push_back("reclaim");check(Json(events)==row.at("events"),"headroom probe order: "+row.dump());
  const double values[]={actual.estimate_gb,actual.free_gb,actual.required_gb,actual.reclaim_gb};for(int i=0;i<4;++i)check(same(values[i],number(row.at("result").at(i))),"headroom result: "+row.dump());
 }
 for(const auto &row:data.at("cleanup")){
  std::vector<std::string> events;double now=0;auto p=probes();p.gpu_count=[]{return 1;};p.now=[&]{return now;};p.free_per_gpu=[&]{events.push_back("total");return std::vector<double>{10.};};p.release_unused_cuda_memory=[&]{events.push_back("collect");events.push_back("cuda_available");if(row.at("cuda").get<bool>())events.push_back("empty_cache");};ResourceManager resources(p);
  for(const auto &model:row.at("models")){now=model.at("last_used");auto t=type(model.at("type"));resources.register_model(t,model.at("type"),"cuda",model.at("vram").get<double>());if(model.at("busy").get<bool>())resources.mark_model_in_use(t,true);}
  reclaim_llm_auxiliary_memory(resources,number(row.at("required")),[&](ModelType t){events.push_back(std::string("destroy:")+model_type_name(t)+":{\"async_cleanup\": false"+(t==ModelType::Embedding?"}":", \"force\": true}"));});
  check(Json(events)==row.at("events"),"cleanup order: "+row.dump()+" actual="+Json(events).dump());
 }
 std::cout<<data.at("headroom").size()<<" source headroom and "<<data.at("cleanup").size()<<" cleanup cases passed\n";
}
struct Gate{std::mutex mutex;std::condition_variable cv;bool entered=false,open=false;void wait(){std::unique_lock<std::mutex> lock(mutex);entered=true;cv.notify_all();cv.wait(lock,[&]{return open;});}void ready(){std::unique_lock<std::mutex>lock(mutex);check(cv.wait_for(lock,std::chrono::seconds(5),[&]{return entered;}),"speech starts");}void release(){std::lock_guard<std::mutex>lock(mutex);open=true;cv.notify_all();}};
void environment(){for(const auto &e:environment_defaults)unsetenv(std::string(e.first).c_str());for(const auto &[k,v]:std::map<std::string,std::string>{{"EZLOCALAI_DOTENV_PATH",""},{"EZLOCALAI_API_KEY","none"},{"DEFAULT_MODEL","first,second,third"},{"LLM_MODEL_RESIDENCY","resident"},{"LLM_MODEL_RESIDENCY_MARGIN_GB","10"},{"LLAMA_CPP_CONTEXT","4096"},{"LLM_MAX_TOKENS","4096"},{"N_PARALLEL","1"},{"LLM_SPECULATIVE_TYPE","none"},{"GPU_LAYERS","0"},{"LLAMA_CPP_START_TIMEOUT","2"},{"LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN},{"LLAMA_CPP_SERVER_URL",""},{"LLAMA_CPP_EMBEDDING_SERVER_URL",""},{"LLAMA_CPP_MODEL_PATHS",""},{"TTS_ENABLED","true"},{"STT_ENABLED","true"},{"TTS_N_PARALLEL","1"},{"STT_N_PARALLEL","1"},{"EMBEDDING_N_PARALLEL","1"},{"EMBEDDING_KEEP_LOADED","true"},{"LAZY_LOAD_VOICE","false"},{"VOICE_SERVER",""},{"TEXT_SERVER",""},{"IMAGE_SERVER",""},{"EMBEDDING_SERVER",""},{"TTS_UNLOAD_LLM_DURING_GENERATION","false"},{"STT_UNLOAD_LLM_DURING_GENERATION","false"},{"IMAGE_ENABLED","false"},{"VIDEO_ENABLED","false"},{"MUSIC_ENABLED","false"}})setenv(k.c_str(),v.c_str(),1);}
void pools(bool busy){
 environment();const auto cwd=fs::current_path();const auto dir=fs::temp_directory_path()/("ezl-headroom-"+std::to_string(getpid())+(busy?"-busy":"-idle"));fs::create_directories(dir);fs::current_path(dir);
 struct Cleanup{fs::path cwd,dir;~Cleanup(){fs::current_path(cwd);fs::remove_all(dir);}}cleanup{cwd,dir};const auto model=dir/"model.gguf";std::ofstream(model)<<"GGUF fixture";
 const int port=fixture_ports(5);setenv("LLAMA_CPP_MODEL_PATH",model.c_str(),1);setenv("LLAMA_CPP_EMBEDDING_MODEL_PATH",model.c_str(),1);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);setenv("LLAMA_CPP_EMBEDDING_SERVER_PORT",std::to_string(port+3).c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",dir.c_str(),1);
 std::atomic<int> cache_clears{0},tts_closed{0},stt_closed{0};std::function<void()> on_cleanup;auto p=probes();p.free_per_gpu=[]{return std::vector<double>{0.};};p.release_unused_cuda_memory=[&]{++cache_clears;if(on_cleanup)on_cleanup();};auto resources=std::make_shared<ResourceManager>(p);
 Gate gate;if(!busy)gate.release();
 TtsPoolFactory tts=[&](auto manager){auto ops=tts_pool_resources({},manager);ops.can_load=[]{return ModelAdmission{true,"cuda","fixture"};};ops.schedule_cleanup=[](auto fn){fn();};ops.load=[&](const auto &){auto m=std::make_shared<TtsPoolModel>();m->device=[]{return "cuda";};m->closed=[]{return false;};m->close=[&]{++tts_closed;};m->generate=[&](const auto &){gate.wait();return "UklGRg==";};return m;};return ops;};
 SttPoolFactory stt=[&](auto manager){auto ops=stt_pool_resources({},manager);ops.can_load=[]{};ops.schedule_cleanup=[](auto fn){fn();};ops.load=[&](const auto &){auto m=std::make_shared<SttPoolModel>();m->device=[]{return "cuda";};m->close=[&]{++stt_closed;};return m;};return ops;};
 auto backend=std::make_unique<LlamaCppInferenceBackend>("first,second,third",resources);auto *engine=backend.get();engine->warm_aux_model("embedding");resources->register_model(ModelType::Embedding,"fixture embedding","cuda",2.);
 ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=3;NativeServer server(config,std::move(backend),resources,stt,tts,{},true);
 const auto chat=[&](const std::string &name){return server.handle_request("POST","/v1/completions","",Json{{"model",name},{"prompt","test"}}.dump());};
 auto speech=[&]{return server.handle_request("POST","/v1/audio/speech","",R"({"input":"hello"})");};
 std::future<HttpResponse> pending;if(busy){pending=std::async(std::launch::async,speech);gate.ready();}struct Release{Gate &g;~Release(){g.release();}}release{gate};
 check(chat("first").status==200,"cold LLM load after auxiliary reclamation");
 check(stt_closed==1&&!resources->is_model_loaded(ModelType::Embedding),"STT and embedding reclaimed synchronously");
 check(tts_closed==(busy?0:1),"active TTS lease protected; idle pool reclaimed");check(cache_clears==1,"global allocator cache cleared once");
 std::ifstream pids(dir/("launch-"+std::to_string(port+3)));int pid;check(bool(pids>>pid)&&kill(pid,0)!=0,"embedding process reaped without backend mutex deadlock");
 auto life=Json::parse(engine->model_lifecycle_json());check(life["vram_estimates_gb"]["first"].get<double>()>.5,"cold request caches target-specific estimate");
 if(busy){gate.release();check(pending.get().status==200,"active speech finishes");}else check(speech().status==200,"speech reloads after reclamation");
 check(resources->is_model_loaded(ModelType::Tts),"speech pool resident again");check(chat("first").status==200&&resources->is_model_loaded(ModelType::Tts)&&cache_clears==1,"resident model reuse does not reclaim auxiliaries");
 check(chat("second").status==200&&!resources->is_model_loaded(ModelType::Tts)&&cache_clears==2,"another cold model triggers new reclamation");
 if(!busy){
  auto detached=engine->unload_llms_for_media();detached(false);resources->register_model(ModelType::Image,"idle fixture","cuda",6.);
  Gate loading;on_cleanup=[&]{loading.wait();};
  auto first=std::async(std::launch::async,[&]{return chat("first");});loading.ready();
  auto second=std::async(std::launch::async,[&]{return chat("second");});struct ReleaseLoads{Gate &gate;~ReleaseLoads(){gate.release();}}release_loads{loading};
  check(second.wait_for(std::chrono::milliseconds(100))==std::future_status::timeout,"second LLM load waits through reclamation");
  check(Json::parse(engine->model_lifecycle_json())["loaded_llms"].empty(),"backend mutex available while allocator cleanup waits");
  loading.release();check(first.get().status==200&&second.get().status==200,"serialized cold loads finish without lock inversion");on_cleanup={};
  check(Json::parse(engine->model_lifecycle_json())["loaded_llms"].size()==2,"resident mode retains both models after serialized loads");
  fs::remove(dir/("seen-"+std::to_string(port)));Gate cold;on_cleanup=[&]{cold.wait();};
  auto active=std::async(std::launch::async,[&]{return server.handle_request("POST","/v1/completions","",R"({"model":"first","prompt":"block"})");});
  for(int n=0;n<1000&&!fs::exists(dir/("seen-"+std::to_string(port)));++n)std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto third=std::async(std::launch::async,[&]{return chat("third");});
  struct Finish{Gate &gate;fs::path release;~Finish(){std::ofstream(release)<<"release";gate.release();}}finish{cold,dir/("release-"+std::to_string(port))};
  cold.ready();std::ofstream(finish.release)<<"release";
  check(active.wait_for(std::chrono::seconds(3))==std::future_status::ready&&active.get().status==200,"completed inference releases its lease while another model reclaims memory");
  check(third.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout,"cold load still waits on cleanup");cold.release();check(third.get().status==200,"cold load resumes after independent inference completion");on_cleanup={};

 }

}
}
int main(int argc,char **argv){try{check(argc==2,"fixture argument");reference(fs::absolute(argv[1]));pools(false);pools(true);std::cout<<"Cold-load pool reclamation, active-lease preservation, allocator release and reuse passed\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
