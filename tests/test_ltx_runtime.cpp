#include "ezlocalai/ltx_runtime.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/process.hpp"
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <csignal>
#include <sys/wait.h>
#include "ltx_fixture_control.hpp"
#define setenv ltx_fixture_setenv
#define unsetenv ltx_fixture_unsetenv
namespace fs=std::filesystem;
using Json=nlohmann::json;
void check(bool b,const std::string &s){if(!b)throw std::runtime_error(s);}
std::string read(const fs::path&p){std::ifstream in(p,std::ios::binary);return {std::istreambuf_iterator<char>(in),{}};}
void write(const fs::path&p,const std::string&s){fs::create_directories(p.parent_path());std::ofstream(p)<<s;}
std::string sha(const std::vector<uint8_t>&bytes){unsigned char d[32];SHA256(bytes.data(),bytes.size(),d);std::string s;for(auto c:d){s+="0123456789abcdef"[c>>4];s+="0123456789abcdef"[c&15];}return s;}
std::vector<Json> captures(const std::string &file="capture.jsonl"){std::ifstream in(file);std::vector<Json>r;for(std::string line;std::getline(in,line);)r.push_back(Json::parse(line));return r;}
std::string arg(const Json &j,const std::string &key){auto a=j.at("args").get<std::vector<std::string>>();for(size_t i=0;i+1<a.size();++i)if(a[i]==key)return a[i+1];return "";}
void clear(){fs::remove("capture.jsonl");}
void seed(){
 const std::string revision(40,'a');
 for(const auto &repo:{"models--unsloth--LTX-2.3-GGUF","models--diffusers--LTX-2.3-Diffusers"})write(fs::path("models")/repo/"refs/main",revision);
 const auto root=fs::path("models/models--diffusers--LTX-2.3-Diffusers/snapshots")/revision;
 write(fs::path("models/models--unsloth--LTX-2.3-GGUF/snapshots")/revision/"ltx-2.3-22b-dev-Q4_K_M.gguf","fixture");
 for(const auto*c:{"vae","audio_vae","vocoder","connectors"})write(root/c/"diffusion_pytorch_model.safetensors","fixture");
 write(root/"text_encoder/model.safetensors.index.json",R"({"weight_map":{"a":"part-1.safetensors","b":"part-2.safetensors"}})");
 write(root/"text_encoder/part-1.safetensors","fixture");write(root/"text_encoder/part-2.safetensors","fixture");write(root/"tokenizer/tokenizer.json","{}");
}
void resize_cases(const fs::path &fixture){
 auto data=Json::parse(read(fixture));
 for(const auto &r:data["cases"]){int w=r["input"][0],h=r["input"][1];ezlocalai::RgbImage image{w,h,{}};for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c)image.pixels.push_back((x*17+y*29+c*71+x*y*3)%256);
  check(sha(ezlocalai::resize_rgb_bilinear(image,r["output"][0],r["output"][1]).pixels)==r["rgb_sha256"],"installed PIL bilinear resize");
 }check(data["cases"].size()==64,"all resize cases");
}
class Tracked:public ezlocalai::InferenceBackend{
public:std::atomic<bool>loaded{true};std::atomic<int>unloads{0},restores{0};
 std::string chat(const std::string&)override{return "{}";}std::string completion(const std::string&s)override{return chat(s);}std::string embedding(const std::string&)override{return "{}";}
 std::string loaded_models_json()const override{return loaded?R"({"llm":{"name":"fixture"}})":"{}";}
 std::function<void(bool)>unload_llms_for_media()override{++unloads;loaded=false;return [this](bool reload){++restores;loaded=reload;};}
};
int text_policy_child(const std::string& device, const std::string& type, int sms, bool fail) {
 char directory[]="/tmp/ezlocalai-ltx-text-policy-XXXXXX";
 check(mkdtemp(directory),"text policy temporary directory");auto previous=fs::current_path();fs::current_path(directory);
 for(const auto&e:ezlocalai::environment_defaults)unsetenv(e.first.data());
 for(const auto* key:{"WAN_FIXTURE_DELAY","WAN_FIXTURE_EXIT","WAN_FIXTURE_INPUT_CAPTURE","LTX_FIXTURE_OOM_COUNT","LTX_FIXTURE_FAIL_AT","VIDEO_CONNECTORS_SOURCE"})unsetenv(key);
 setenv("HF_HUB_OFFLINE","1",1);setenv("HF_HUB_CACHE","models",1);setenv("SDCPP_BIN",TEST_LTX_FIXTURE_BIN,1);
 setenv("LTX_FIXTURE_CAPTURE","capture.jsonl",1);setenv("VIDEO_GPU_RESIDENCY","model_offload",1);
 if(fail)setenv("LTX_FIXTURE_NF4_FAIL","1",1);else unsetenv("LTX_FIXTURE_NF4_FAIL");
 seed();ezlocalai::LtxVideo runtime("unsloth/LTX-2.3-GGUF",device);ezlocalai::LtxVideoSettings settings;settings.size="64x64";settings.frames=9;settings.steps=2;
 check(fs::is_regular_file(runtime.generate("text encoder policy",settings,"")),"text policy process output");
 const auto launches=captures("preload.jsonl");check(launches.size()==(device=="cpu"?1:2)+(fail&&sms?1:0),"initial load, post-load placement and clean NF4 retry");
 check(captures("contexts.jsonl").size()==1,"placement reuses the successful context");check(captures().size()==1,"generation uses the prepared context");
 check(arg(launches.front(),"--tensor-type-rules")=="^text_encoders.llm.*="+type,"source compute capability selects precision");
 check(arg(launches.front(),"--ltx-nf4-sm-count")== (sms?std::to_string(sms):""),"source 12 GiB threshold selects NF4");
 if(sms)check(arg(launches.front(),"--ltx-nf4-threads-per-sm")=="1536","selected GPU geometry reaches child");
 if(fail&&sms)check(arg(launches.back(),"--ltx-nf4-sm-count").empty()&&arg(launches.front(),"-o")!=arg(launches.back(),"-o"),"fallback drops NF4 and uses fresh working files");
 fs::current_path(previous);fs::remove_all(directory);return 0;
}
void text_policy_cases(const std::string& self) {
 const char* previous=std::getenv("LD_LIBRARY_PATH");const std::string old=previous?previous:"";
 setenv("LD_LIBRARY_PATH",(std::string(TEST_CUDA_DRIVER_DIRECTORY)+(old.empty()?"":":"+old)).c_str(),1);
 const auto inventory=fs::absolute("text-devices.tsv");setenv("EZLOCALAI_TEST_CUDA_DEVICES",inventory.c_str(),1);unsetenv("CUDA_VISIBLE_DEVICES");unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
 for(int major:{7,8})for(uint64_t free:{(12ULL<<30)-1,12ULL<<30})for(bool fail:{false,true}) {
  write(inventory,"NVIDIA fixture\t"+std::to_string(major)+"\t0\t25769803776\t"+std::to_string(free)+"\n");
  const auto result=ezlocalai::run_process({self,"--text-policy","cuda:0",major>=8?"bf16":"f16",free>=(12ULL<<30)?std::to_string(major*10):"0",fail?"1":"0"},true);
  check(result.status==0,"GPU text policy/fallback child: "+result.output);
 }
 auto result=ezlocalai::run_process({self,"--text-policy","cpu","bf16","0","0"},true);check(result.status==0,"CPU text policy: "+result.output);
 setenv("EZLOCALAI_TEST_CUDA_FAIL","attribute",1);
 result=ezlocalai::run_process({self,"--text-policy","cuda:0","bf16","0","0"},true);check(result.status==0,"attribute failure selects unquantized fallback: "+result.output);
 unsetenv("EZLOCALAI_TEST_CUDA_FAIL");unsetenv("EZLOCALAI_TEST_CUDA_DEVICES");
 if(previous)setenv("LD_LIBRARY_PATH",old.c_str(),1);else unsetenv("LD_LIBRARY_PATH");
}
// These source-oracle cases validate placement decisions and context ownership.
// Quanto tensor execution and Python's swallowed sequential-hook errors are not
// represented by this synthetic engine and remain separate parity gaps.
int placement_child(const fs::path& fixture,size_t index) {
 const auto row=Json::parse(read(fixture)).at("cases").at(index),input=row.at("input"),expected=row.at("result");
 char directory[]="/tmp/ezlocalai-ltx-placement-XXXXXX";check(mkdtemp(directory),"placement temporary directory");
 const auto previous=fs::current_path();fs::current_path(directory);
 for(const auto&e:ezlocalai::environment_defaults)unsetenv(e.first.data());
 for(const auto* key:{"LTX_FIXTURE_NF4_FAIL","LTX_FIXTURE_PRELOAD_FAIL","LTX_FIXTURE_PLACEMENT_FAIL","VIDEO_GPU_RESIDENCY"})unsetenv(key);
 setenv("HF_HUB_OFFLINE","1",1);setenv("HF_HUB_CACHE","models",1);setenv("SDCPP_BIN",TEST_LTX_FIXTURE_BIN,1);
 setenv("LTX_FIXTURE_CAPTURE","capture.jsonl",1);
 if(input.contains("env"))for(const auto& [key,value]:input.at("env").items())setenv(key.c_str(),value.get<std::string>().c_str(),1);
 const auto device=input.value("device",std::string("cuda:1"));const int major=input.value("major",8);
 const auto ordinal=device=="cpu"?1:std::stoi(device.substr(5));
 auto inventory=[&](double free){std::string out;for(int i=0;i<3;++i)out+="NVIDIA fixture\t"+std::to_string(major)+"\t0\t"+std::to_string(uint64_t(input.at("total").get<double>()*(1ULL<<30)))+"\t"+std::to_string(i==ordinal?uint64_t(free*(1ULL<<30)):0)+"\n";return out;};
 const auto inventory_path=fs::absolute("devices.tsv");write(inventory_path,inventory(input.at("before")));
 setenv("EZLOCALAI_TEST_CUDA_DEVICES",inventory_path.c_str(),1);
 setenv("LTX_FIXTURE_AFTER_MEMORY",inventory(input.at("after")).c_str(),1);
 if(input.contains("fail")){
  const auto failures=input.at("fail").get<std::vector<std::string>>();std::string placement;
  for(const auto& failure:failures){if(failure=="nf4")setenv("LTX_FIXTURE_NF4_FAIL","1",1);else if(failure!="quanto")placement+=failure+",";}
  if(!placement.empty())setenv("LTX_FIXTURE_PLACEMENT_FAIL",placement.c_str(),1);
 }
 std::optional<ezlocalai::LtxVideoSettings> hint;
 if(input.contains("hint")&&!input.at("hint").empty()){
  const auto& source=input.at("hint");hint.emplace();hint->frames=source.at("num_frames");hint->size=std::to_string(source.at("width").get<int>())+"x"+std::to_string(source.at("height").get<int>());
 }
 seed();{
  ezlocalai::LtxVideo runtime("unsloth/LTX-2.3-GGUF",device,hint);
  check(runtime.loaded()&&runtime.residency()==expected.at("residency"),"post-load source placement case "+std::to_string(index));
  const auto launches=captures("preload.jsonl");check(!launches.empty(),"placement loaded engine");
  check(arg(launches.front(),"--tensor-type-rules")=="^text_encoders.llm.*="+(expected.at("dtype")=="bf16"?std::string("bf16"):std::string("f16")),"placement selected GPU precision");
  const bool quantized=device!="cpu"&&input.at("before").get<double>()>=12;
  check(arg(launches.front(),"--ltx-nf4-sm-count")== (quantized?std::to_string(major*10):""),"before-load memory selects encoder");
  if(device!="cpu")check(arg(launches.back(),"--ltx-offload-mode")== (runtime.residency()=="sequential"?"2":"1"),"selected execution mode reaches retained engine");
  if(!input.contains("fail")&&runtime.residency()!="full")check(captures("contexts.jsonl").size()==1,"post-load placement preserves Gemma context");
 }
 const auto contexts=captures("contexts.jsonl");size_t loads=0,frees=0;for(const auto& event:contexts){loads+=event.at("event")=="load";frees+=event.at("event")=="free";}
 check(loads==frees,"every placement context released");
 fs::current_path(previous);fs::remove_all(directory);return 0;
}
void placement_cases(const std::string& self,const fs::path& fixture){
 const std::string old=std::getenv("LD_LIBRARY_PATH")?std::getenv("LD_LIBRARY_PATH"):"";
 setenv("LD_LIBRARY_PATH",(std::string(TEST_CUDA_DRIVER_DIRECTORY)+(old.empty()?"":":"+old)).c_str(),1);
 unsetenv("CUDA_VISIBLE_DEVICES");unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
 size_t count=0;const auto rows=Json::parse(read(fixture)).at("cases");
 for(size_t index=0;index<rows.size();++index){
  if(index==41||index==42)continue; // Source ignores failed sequential-hook installation.
  const auto result=ezlocalai::run_process({self,"--placement",fixture.string(),std::to_string(index)},true);
  check(result.status==0,"source placement child "+std::to_string(index)+": "+result.output);++count;
 }
 if(old.empty())unsetenv("LD_LIBRARY_PATH");else setenv("LD_LIBRARY_PATH",old.c_str(),1);
 std::cout<<count<<" source constructor placement cases passed (tensor execution excluded)\n";
}
void lifetime_cases(const fs::path &fixture) {
 const auto rows=Json::parse(read(fixture)).at("cases");
 for(const auto &row:rows) {
  clear();fs::remove("preload.jsonl");fs::remove("contexts.jsonl");
  setenv("MEDIA_KEEP_MODELS_LOADED",row.at("keep").get<bool>()?"true":"false",1);
  setenv("VIDEO_UNLOAD_LLM_DURING_GENERATION",row.at("mode").get<std::string>().c_str(),1);
  setenv("DEFAULT_MODEL",row.at("resident").get<bool>()?"fixture":"none",1);
  auto tracked=std::make_unique<Tracked>();auto *backend=tracked.get();backend->loaded=row.at("resident");
  ezl_env_config config{};ezl_load_config(&config);auto resources=std::make_shared<ezlocalai::ResourceManager>();
  std::vector<int> pids;
  {
   ezlocalai::NativeServer server(config,std::move(tracked),resources);
   if(row.at("preload").get<bool>())server.preload_media();
   for(size_t i=0;i<2;++i) {
    const Json request={{"prompt",i?"changed prompt":"first prompt"},{"size",i?"96x64":"65x95"},{"num_frames",i?17:10},{"num_inference_steps",2},{"response_format","url"}};
    const auto reply=server.handle_request("POST","/v1/videos/generations","",request.dump());
    check(reply.status==200,"retained LTX HTTP generation: "+reply.body);
    const auto &expected=row.at("snapshots").at(i);
    const auto contexts=captures("contexts.jsonl");
    check(std::count_if(contexts.begin(),contexts.end(),[](const Json& event){return event.at("event")=="load";})==expected.at("loads").get<int>(),"source number of pipeline loads");
    check(resources->is_model_loaded(ezlocalai::ModelType::Video)==expected.at("registered").get<bool>(),"source retained registration");
    check(resources->get_model_active_count(ezlocalai::ModelType::Video)==expected.at("active").get<int>(),"no phantom video usage after completion");
    auto published=Json::parse(server.handle_request("GET","/v1/resources","","").body);
    check(published.at("loaded_models").contains("video")==expected.at("registered").get<bool>(),"published retained video presence");
   }
   auto loads=captures("contexts.jsonl");
   loads.erase(std::remove_if(loads.begin(),loads.end(),[](const Json& event){return event.at("event")!="load";}),loads.end());
   const auto generations=captures();
   for(const auto &load:loads)pids.push_back(load.at("pid"));
   check(generations.size()==2,"two source generation operations");
   for(size_t i=0;i<2;++i)check(generations[i].at("pid")==pids.at(row.at("generations").at(i).get<size_t>()),"source context reuse across changed requests");
   check(captures("preload.jsonl").size()>=loads.size(),"every context is preloaded before generation");
   const auto events=row.at("events").get<std::vector<std::string>>();
   check(backend->unloads==std::count(events.begin(),events.end(),"unload_llm")&&backend->restores==std::count(events.begin(),events.end(),"restore_llm"),"source LLM handoff counts");
  }
  for(int pid:pids) {check(kill(pid,0)==-1&&errno==ESRCH,"retained engine reaped at teardown");check(waitpid(pid,nullptr,WNOHANG)==-1&&errno==ECHILD,"no zombie retained engine");}
 }
 // A worker failure invalidates the registration; the next request preloads
 // a fresh context before admitting generation. Failed startup is retryable too.
 setenv("DEFAULT_MODEL","none",1);setenv("VIDEO_UNLOAD_LLM_DURING_GENERATION","false",1);
 clear();fs::remove("preload.jsonl");fs::remove("contexts.jsonl");
 ezl_env_config config{};ezl_load_config(&config);auto backend=std::make_unique<Tracked>();backend->loaded=false;
 auto resources=std::make_shared<ezlocalai::ResourceManager>();
 ezlocalai::NativeServer server(config,std::move(backend),resources);
 setenv("LTX_FIXTURE_PRELOAD_FAIL","1",1);server.preload_media();
 check(!resources->is_model_loaded(ezlocalai::ModelType::Video),"failed startup is not registered");unsetenv("LTX_FIXTURE_PRELOAD_FAIL");
 auto generate=[&]{return server.handle_request("POST","/v1/videos/generations","",R"({"prompt":"recover","size":"64x64","num_frames":9,"num_inference_steps":2})");};
 check(generate().status==200&&resources->is_model_loaded(ezlocalai::ModelType::Video),"lazy recovery after startup failure");
 const int first=captures().back().at("pid");
 setenv("WAN_FIXTURE_EXIT","7",1);check(generate().status==500,"inference failure returned");unsetenv("WAN_FIXTURE_EXIT");
 check(!resources->is_model_loaded(ezlocalai::ModelType::Video)&&resources->get_model_active_count(ezlocalai::ModelType::Video)==0,"dead engine removed from registry");
 check(!Json::parse(server.handle_request("GET","/v1/resources","","").body).at("loaded_models").contains("video"),"dead engine absent from published models");
 check(kill(first,0)==-1&&errno==ESRCH,"failed engine reaped before retry");
 check(generate().status==200&&resources->is_model_loaded(ezlocalai::ModelType::Video)&&captures().back().at("pid")!=first,"fresh engine after generation failure");
 const int recovered=captures().back().at("pid");
 setenv("LTX_FIXTURE_MALFORMED_REPLY","1",1);check(generate().status==500,"malformed private reply fails request");unsetenv("LTX_FIXTURE_MALFORMED_REPLY");
 check(!resources->is_model_loaded(ezlocalai::ModelType::Video)&&kill(recovered,0)==-1&&errno==ESRCH,"protocol parser exception removes and reaps engine");
 check(generate().status==200&&resources->is_model_loaded(ezlocalai::ModelType::Video),"fresh engine after protocol parser exception");
 unsetenv("MEDIA_KEEP_MODELS_LOADED");unsetenv("VIDEO_UNLOAD_LLM_DURING_GENERATION");
 std::cout<<rows.size()<<" source LTX lifetime cases and failed-startup/inference recovery passed\n";
}
int real_placement_case(const fs::path& path) {
 const auto config=Json::parse(read(path));
 for(const auto& [key,value]:config.at("environment").items())::setenv(key.c_str(),value.get<std::string>().c_str(),1);
 ezlocalai::LtxVideoSettings settings;settings.size="64x64";settings.frames=9;settings.steps=2;
 Json report;std::vector<pid_t> children;
 auto child=[] {const auto value=read("/proc/self/task/"+std::to_string(getpid())+"/children");check(!value.empty(),"real retained engine child");return pid_t(std::stoi(value));};
 auto phase=[&](const std::string& name){std::ofstream("observation.json.tmp")<<Json{{"phase",name}}.dump();fs::rename("observation.json.tmp","observation.json");};
 const auto start=std::chrono::steady_clock::now();
 {
  ezlocalai::LtxVideo runtime(config.at("model"),config.at("device"),settings);
  report["preload_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  report["initial_residency"]=runtime.residency();children.push_back(child());report["initial_pid"]=children.back();
  check(runtime.loaded()&&runtime.residency()==config.at("expected_initial"),"real post-load placement");
  phase("preloaded");std::this_thread::sleep_for(std::chrono::seconds(3));
  report["requests"]=Json::array();
  for(int i=0;i<2;++i){
   phase("request-"+std::to_string(i+1)+"-running");const auto begin=std::chrono::steady_clock::now();
   const auto output=runtime.generate("A red ball rolls slowly.",settings,"");
   check(!output.empty()&&fs::is_regular_file(output),"real LTX output after placement recovery");
   children.push_back(child());
   report["requests"].push_back({{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()},
       {"pid",children.back()},{"output",output},{"residency",runtime.residency()}});
   check(runtime.loaded()&&runtime.residency()==config.at("expected_final"),"real generation fallback placement");
   phase("request-"+std::to_string(i+1)+"-complete");std::this_thread::sleep_for(std::chrono::seconds(3));
  }
  check(children[1]==children[2],"successful recovery retains engine for next generation");
  if(config.value("expect_restart",false))check(children[0]!=children[1],"failed whole-model execution restarted before sequential retry");
  else check(children[0]==children[1],"successful whole-model execution retains initial engine");
  phase("stopping");
 }
 for(auto pid:children)check(kill(pid,0)<0&&errno==ESRCH,"every real engine reaped");
 report["reaped"]=true;std::ofstream("result.json")<<report.dump(2)<<'\n';std::cout<<report.dump()<<'\n';return 0;
}
int main(int argc,char**argv){try{
 if(argc==3&&std::string(argv[1])=="--real-placement")return real_placement_case(argv[2]);
 if(argc==4&&std::string(argv[1])=="--placement")return placement_child(argv[2],std::stoul(argv[3]));
 if(argc==6&&std::string(argv[1])=="--text-policy")return text_policy_child(argv[2],argv[3],std::stoi(argv[4]),std::string(argv[5])=="1");
 check(argc==2,"resize fixture required");auto fixture=fs::absolute(argv[1]);const auto self=fs::absolute(argv[0]).string();auto previous=fs::current_path();char dir[]="/tmp/ezlocalai-ltx-runtime-XXXXXX";check(mkdtemp(dir),"temp directory");fs::current_path(dir);
 for(const auto&e:ezlocalai::environment_defaults)unsetenv(e.first.data());
 for(const auto*k:{"VIDEO_GPU_RESIDENCY","VIDEO_CONNECTORS_SOURCE","VIDEO_GENERATION_TIMEOUT","HF_ENDPOINT","HF_HUB_CACHE","LTX2_CONFIG_REPO","WAN_FIXTURE_EXIT","WAN_FIXTURE_DELAY","WAN_FIXTURE_MISSING_FRAME","WAN_FIXTURE_ARGUMENTS","GGML_CUDA_NO_PINNED"})unsetenv(k);
 setenv("HF_HUB_OFFLINE","1",1);setenv("SDCPP_BIN",TEST_LTX_FIXTURE_BIN,1);setenv("LTX_FIXTURE_CAPTURE","capture.jsonl",1);
 setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("ROUTER_DRAIN_FILE","",1);setenv("USAGE_FILE","usage.json",1);setenv("DEFAULT_MODEL","none",1);setenv("VIDEO_ENABLED","true",1);setenv("VIDEO_MODEL","unsloth/LTX-2.3-GGUF",1);setenv("EZLOCALAI_URL","",1);
 seed();resize_cases(fixture);auto assets=ezlocalai::prepare_ltx_assets("unsloth/LTX-2.3-GGUF");check(assets.components.find(std::string(40,'a'))!=std::string::npos,"reuse Python refs/main snapshot");
 fs::remove(fs::path(assets.components)/"text_encoder/part-2.safetensors");bool failed=false;try{ezlocalai::prepare_ltx_assets("unsloth/LTX-2.3-GGUF");}catch(const std::exception&){failed=true;}check(failed,"incomplete cached shards rejected");seed();
 ezlocalai::LtxVideo runtime("unsloth/LTX-2.3-GGUF","cuda:3");ezlocalai::LtxVideoSettings s;s.size="65x95";s.frames=10;s.steps=1;
 auto output=runtime.generate("literal $(touch unsafe)",s,"");check(fs::file_size(output)>0&&!fs::exists("unsafe"),"native process output, no shell");
 auto launch=captures().back();check(arg(launch,"--backend")=="cuda3"&&arg(launch,"--steps")=="2"&&arg(launch,"--video-frames")=="9"&&launch["pageable"]=="1"&&!std::getenv("GGML_CUDA_NO_PINNED"),"GPU/settings/child-only environment");
 check(arg(launch,"--extra-sample-args")==ezlocalai::ltx_reference_scheduler_args,"fixed reference scheduler shift independent of clip size");
 check(arg(launch,"--rng")=="cpu"&&arg(launch,"--seed")=="42","reference CPU generator and fixed seed on CUDA");
 auto probe=ezlocalai::run_process({"ffprobe","-v","error","-show_streams","-of","json",output},true);auto streams=Json::parse(probe.output)["streams"];check(streams.size()==2&&streams[0]["nb_frames"]=="9"&&streams[1]["sample_rate"]=="24000"&&streams[1]["duration"]=="0.330000","audio resampling preserves the 0.33-second vocoder waveform");
 // The fixture supplies a 480 Hz stereo sawtooth at 48 kHz. Relabeling it
 // as 24 kHz halves its pitch; duration alone would miss a later tempo fix.
 auto pcm=ezlocalai::run_process({"ffmpeg","-v","error","-i",output,"-vn","-ac","1","-ar","24000","-f","s16le","-"},true);
 check(pcm.status==0&&pcm.output.size()>=2*6720,"decode exported audio");
 bool negative=false;size_t crossings=0;
 for(size_t i=1200;i<6720;++i){
  int value=static_cast<unsigned char>(pcm.output[2*i])|(static_cast<unsigned char>(pcm.output[2*i+1])<<8);
  if(value>=32768)value-=65536;
  if(value < -400)negative=true;
  if(value > 400&&negative){++crossings;negative=false;}
 }
 const double frequency=crossings/(5520.0/24000);
 check(std::abs(frequency-480)<20,"audio resampling preserves vocoder pitch");
 check(!fs::exists(fs::path(arg(launch,"-o")).parent_path()),"temporary frames cleaned");
 const auto silent=runtime.generate("silent",s,"","",{},false);
 auto silent_probe=ezlocalai::run_process({"ffprobe","-v","error","-show_streams","-of","json",silent},true);
 check(Json::parse(silent_probe.output)["streams"].size()==1,"internal include_audio=false exports video only");
 setenv("LTX_FIXTURE_NO_AUDIO","1",1);const auto missing_audio=runtime.generate("no audio",s,"");unsetenv("LTX_FIXTURE_NO_AUDIO");
 silent_probe=ezlocalai::run_process({"ffprobe","-v","error","-show_streams","-of","json",missing_audio},true);
 check(Json::parse(silent_probe.output)["streams"].size()==1,"missing audio still exports video");
 ezlocalai::RgbImage image{3,5,std::vector<uint8_t>(45,100)};auto png=ezlocalai::encode_rgb_png(image);
 clear();runtime.generate("condition",s,"",png,{{png,0,1},{png,-1,.25}});launch=captures().back();check(launch["conditions"].size()==2&&launch["conditions"][1]["index"]==-1&&launch["conditions"][1]["strength"]==.25&&launch["conditions"][0]["exists"]==true&&arg(launch,"-i").empty(),"decoded conditions with priority and individual strength");
 clear();runtime.generate("image",s,"",png);launch=captures().back();check(!arg(launch,"-i").empty()&&arg(launch,"--strength")=="1","I2V mode");
 clear();setenv("VIDEO_GPU_RESIDENCY","full",1);setenv("LTX_FIXTURE_OOM_COUNT","2",1);{ ezlocalai::LtxVideo retry("unsloth/LTX-2.3-GGUF","cuda:3");retry.generate("retry",s,"",png,{{png,1,.5}}); }auto attempts=captures();check(attempts.size()==3&&arg(attempts[2],"--ltx-conditions").empty()&&arg(attempts[2],"-i").empty(),"sequential then reduced retry drops conditioning");unsetenv("LTX_FIXTURE_OOM_COUNT");unsetenv("VIDEO_GPU_RESIDENCY");
 clear();setenv("WAN_FIXTURE_EXIT","7",1);check(runtime.generate("failure",s,"").empty()&&captures().size()==1,"non-CUDA failure not retried");unsetenv("WAN_FIXTURE_EXIT");
 clear();setenv("WAN_FIXTURE_DELAY","1000",1);setenv("VIDEO_GENERATION_TIMEOUT","0.02",1);failed=false;try{runtime.generate("timeout",s,"");}catch(const std::runtime_error&){failed=true;}{const auto timed_out=captures();check(failed&&!timed_out.empty()&&!fs::exists(fs::path(arg(timed_out.back(),"-o")).parent_path()),"timeout reaps process and removes working files");}unsetenv("WAN_FIXTURE_DELAY");unsetenv("VIDEO_GENERATION_TIMEOUT");
 clear();setenv("DEFAULT_MODEL","fixture",1);setenv("VIDEO_UNLOAD_LLM_DURING_GENERATION","true",1);setenv("WAN_FIXTURE_DELAY","250",1);
 ezl_env_config config{};ezl_load_config(&config);auto tracked=std::make_unique<Tracked>();auto*p=tracked.get();ezlocalai::NativeServer server(config,std::move(tracked));
 const auto request=R"({"prompt":"test","size":"64x64","num_frames":9,"num_inference_steps":2,"response_format":"b64_json","n":2})";
 auto pending=std::async(std::launch::async,[&]{return server.handle_request("POST","/v1/videos/generations","",request);});
 for(int i=0;i<500&&!fs::exists("capture.jsonl");++i)std::this_thread::sleep_for(std::chrono::milliseconds(10));
 const auto resource=server.handle_request("GET","/v1/resources","","");check(resource.body.find("\"video\":{\"capacity\":1,\"in_flight\":1")!=std::string::npos,"LTX active resources");
 check(server.handle_request("POST","/v1/chat/completions","",R"({"messages":[]})").body.find("temporarily_unavailable")!=std::string::npos,"LLM admission reserved");
 auto response=pending.get();check(response.status==200&&Json::parse(response.body)["data"].size()==2&&p->loaded&&p->unloads==2&&p->restores==2,"HTTP base64 n and per-item LLM restoration: "+response.body.substr(0,150));unsetenv("WAN_FIXTURE_DELAY");
 setenv("WAN_FIXTURE_EXIT","3",1);response=server.handle_request("POST","/v1/videos/generations","",request);check(response.status==500&&p->loaded&&p->unloads==3&&p->restores==3,"HTTP failure restores LLM");
#ifdef EZLOCALAI_STORYBOARD
 unsetenv("WAN_FIXTURE_EXIT");setenv("MUSIC_ENABLED","true",1);
 fs::create_directories("outputs");
 check(ezlocalai::run_process({"ffmpeg","-v","error","-y","-f","lavfi","-i","sine=frequency=440:sample_rate=24000","-t","1.2","outputs/song.wav"},true).status==0,"music-video audio fixture");
 Json song={{"prompt","Pythagorean concert"},{"lyrics","a squared plus b squared equals c squared"},{"keyscale","C minor"},
  {"duration",1.2},{"scene_duration",1.0},{"audio_url","outputs/song.wav"},{"size","64x64"},{"frame_rate",8},{"num_inference_steps",2}};
 auto music_video=[&](const Json&r,const std::string&path="/v1/videos/music"){return server.handle_request("POST",path,"",r.dump());};
 clear();setenv("WAN_FIXTURE_DELAY","150",1);
 auto music_pending=std::async(std::launch::async,[&]{return music_video(song);});
 for(int i=0;i<500&&!fs::exists("capture.jsonl");++i)std::this_thread::sleep_for(std::chrono::milliseconds(10));
 check(server.handle_request("POST","/v1/chat/completions","",R"({"messages":[]})").body.find("temporarily_unavailable")!=std::string::npos,"music-video reserves LLM admission");
 response=music_pending.get();unsetenv("WAN_FIXTURE_DELAY");
 check(response.status==200,"music-video HTTP: "+response.body);
 auto item=Json::parse(response.body)["data"][0];
 check(item["scene_count"]==2&&item["math_overlay"]==true&&item["storyboard"]==true&&item["scene_duration"]==1.0,"music-video metadata");
 check(p->loaded&&p->unloads==4&&p->restores==4&&captures().size()==2,"one LLM handoff across two scenes");
 for(const auto&launch:captures())check(!arg(launch,"-i").empty(),"default storyboard reaches each engine invocation");
 for(const auto&launch:captures())check(arg(launch,"--extra-sample-args")==ezlocalai::ltx_reference_scheduler_args,"reference schedule reaches each scene");
 auto mux=ezlocalai::run_process({"ffprobe","-v","error","-show_streams","-show_format","-of","json",item["url"]},true);
 auto media=Json::parse(mux.output);check(media["streams"].size()==2&&std::abs(std::stod(media["format"]["duration"].get<std::string>())-1.2)<.15,"final song audio and trimmed video");
 for(const auto&url:item["scene_urls"]){auto probe=ezlocalai::run_process({"ffprobe","-v","error","-show_streams","-of","json",url},true);check(Json::parse(probe.output)["streams"].size()==1,"scene audio disabled by default");}
 Json supplied=song;supplied["prompt"]="Concert";supplied["lyrics"]="Dance tonight";supplied["duration"]=.2;supplied["include_scene_audio"]=true;supplied["response_format"]="b64_json";
 auto encoded=std::string(4*((png.size()+2)/3),'\0');EVP_EncodeBlock(reinterpret_cast<unsigned char*>(encoded.data()),reinterpret_cast<const unsigned char*>(png.data()),png.size());
 supplied["scene_images"]=Json::array({"",encoded});
 clear();setenv("WAN_FIXTURE_INPUT_CAPTURE","selected.png",1);
 response=music_video(supplied,"/v1/videos/music/generations");
 check(response.status==200,"music-video alias/base64/supplied-image: "+response.body);
 item=Json::parse(response.body)["data"][0];
 check(item.contains("b64_json")&&!item.contains("url")&&item["math_overlay"]==false&&fs::is_regular_file("selected.png"),"base64 and filtered scene image");
 check(ezlocalai::decode_rgb_image(read("selected.png")).pixels==ezlocalai::resize_rgb_bilinear(image,64,64).pixels,"falsey scene image removed before indexing");
 {auto probe=ezlocalai::run_process({"ffprobe","-v","error","-show_streams","-of","json",item["scene_urls"][0]},true);check(Json::parse(probe.output)["streams"].size()==2,"include_scene_audio forwarded");}
 unsetenv("WAN_FIXTURE_INPUT_CAPTURE");
 const auto before=p->unloads.load();
 Json bad=song;bad["image"]="not base64";response=music_video(bad);check(response.status==500&&p->unloads==before,"image decode fails before handoff");
 bad=song;bad["scene_images"]=Json::array({encoded,"invalid"});bad["conditions"]=Json::array({{{"image",encoded}}});
 response=music_video(bad);check(response.status==500&&p->unloads==before,"decode overridden and unused scene images before generation");
 bad=song;bad["response_format"]="wrong";response=music_video(bad);check(response.status==400&&p->unloads==before,"format validation precedes handoff");
 bad=song;bad["audio_url"]="outside.wav";response=music_video(bad);check(response.status==400&&p->unloads==before,"output containment precedes handoff");
 bad=song;bad["size"]="invalid";response=music_video(bad);check(response.status==400&&p->loaded&&p->unloads==before+1&&p->restores==before+1,"invalid scene settings restore LLM");
 clear();setenv("WAN_FIXTURE_EXIT","3",1);response=music_video(song);unsetenv("WAN_FIXTURE_EXIT");
 check(response.status==502&&response.body.find("Video scene 1 failed to generate")!=std::string::npos&&p->loaded&&p->unloads==before+2&&p->restores==before+2,"failed scene restores LLM and maps 502");
 clear();Json conditioned=supplied;conditioned["conditions"]=Json::array({{{"image",encoded},{"index",0},{"strength",.25}}});response=music_video(conditioned);
 check(response.status==200&&captures().size()==1&&arg(captures()[0],"-i").empty()&&captures()[0]["conditions"][0]["strength"]==.25,"positioned conditions override valid scene images");
 check(p->loaded&&p->unloads==before+3&&p->restores==before+3,"admission and handoff recover after failures");
 clear();setenv("LTX_FIXTURE_FAIL_AT","1",1);response=music_video(song);unsetenv("LTX_FIXTURE_FAIL_AT");
 check(response.status==502&&response.body.find("Video scene 2 failed to generate")!=std::string::npos&&captures().size()==2&&p->loaded&&p->unloads==before+4&&p->restores==before+4,"second scene failure still restores exactly once");
 write("outputs/invalid.wav","invalid audio");bad=song;bad["audio_url"]="outputs/invalid.wav";response=music_video(bad);
 check(response.status==502&&p->loaded&&p->unloads==before+5&&p->restores==before+5,"stitch failure occurs after model restoration");
 bad=song;bad["image"]="";clear();response=music_video(bad);
 check(response.status==502&&captures().empty()&&p->loaded&&p->unloads==before+6&&p->restores==before+6,"explicit empty image selects invalid I2V instead of a default storyboard");
 setenv("IMAGE_SERVER","http://127.0.0.1:1",1);clear();response=music_video(song);unsetenv("IMAGE_SERVER");
 check(response.status==200&&Json::parse(response.body)["data"][0].contains("error")&&captures().empty()&&p->loaded&&p->unloads==before+6,"image-server configuration delegates music-video without a local handoff");


#endif
 text_policy_cases(self);
 placement_cases(self,fixture.parent_path()/"ltx_placement_cases.json");
 unsetenv("IMAGE_SERVER");unsetenv("MUSIC_ENABLED");unsetenv("WAN_FIXTURE_EXIT");
 lifetime_cases(fixture.parent_path()/"ltx_lifetime_cases.json");
 fs::current_path(previous);fs::remove_all(dir);std::cout<<"Verified 64 installed resize cases, LTX cache/process/encoding/retry, text quantization policy and HTTP handoff\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
