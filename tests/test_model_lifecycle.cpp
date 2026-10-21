#include "ezlocalai/model_lifecycle.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace fs=std::filesystem;
namespace {
void expect(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
std::string read(const fs::path &path){std::ifstream in(path);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
std::vector<std::string> split(const std::string &value,char separator='\t'){std::vector<std::string> out;size_t pos=0;for(;;){const auto end=value.find(separator,pos);out.push_back(value.substr(pos,end==std::string::npos?end:end-pos));if(end==std::string::npos)return out;pos=end+1;}}
std::string unhex(const std::string &s){std::string out;for(size_t i=0;i<s.size();i+=2)out+=static_cast<char>(std::stoi(s.substr(i,2),nullptr,16));return out;}
std::string sha256(const std::string &value){unsigned char digest[32];SHA256(reinterpret_cast<const unsigned char *>(value.data()),value.size(),digest);std::ostringstream out;for(auto c:digest)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return out.str();}
bool same(double value,const std::string &expected){const auto n=std::strtod(expected.c_str(),nullptr);return value==n||(std::isnan(value)&&std::isnan(n));}
size_t reference_cases(const fs::path &source,const fs::path &fixture){
    size_t count=0;std::istringstream in(read(fixture));
    for(std::string line;std::getline(in,line);){
        if(line.empty())continue;
        if(line[0]=='#'){const auto marker=line.find(" sha256=");if(marker!=std::string::npos)expect(sha256(read(source/line.substr(2,marker-2)))==line.substr(marker+8),"lifecycle source changed");continue;}
        ++count;const auto c=split(line);double clock=0;ezlocalai::ModelLifecycle lifecycle([&]{return clock;});
        if(!c[1].empty())for(const auto &row:split(unhex(c[1]),'\n')){const auto op=split(row);clock=std::strtod(op[4].c_str(),nullptr);lifecycle.record(unhex(op[0]),unhex(op[1]),unhex(op[2]),std::strtod(op[3].c_str(),nullptr),op[5]=="1");}
        const auto actual=lifecycle.snapshot();size_t index=0;
        if(!c[2].empty())for(const auto &row:split(unhex(c[2]),'\n')){const auto v=split(row);expect(index<actual.summary.size(),"missing lifecycle summary");const auto &s=actual.summary[index++];
            expect(s.operation==unhex(v[0])&&s.model_type==unhex(v[1])&&s.model==unhex(v[2])&&s.count==std::stoull(v[3])&&s.failures==std::stoull(v[4])&&same(s.min_seconds,v[5])&&same(s.max_seconds,v[6])&&same(s.last_seconds,v[7])&&same(s.average_seconds,v[8]),"source lifecycle summary case "+c[0]+" row "+std::to_string(index));}
        expect(index==actual.summary.size(),"extra lifecycle summary");index=0;
        if(!c[3].empty())for(const auto &row:split(unhex(c[3]),'\n')){const auto v=split(row);expect(index<actual.recent_events.size(),"missing lifecycle event");const auto &e=actual.recent_events[index++];
            expect(same(e.timestamp,v[0])&&e.operation==unhex(v[1])&&e.model_type==unhex(v[2])&&e.model==unhex(v[3])&&same(e.duration_seconds,v[4])&&e.success==(v[5]=="1"),"source lifecycle event case "+c[0]+" row "+std::to_string(index));}
        expect(index==actual.recent_events.size(),"extra lifecycle event");
    }
    expect(count==818,"all lifecycle cases");return count;
}
void concurrency(){
    ezlocalai::ModelLifecycle lifecycle;std::vector<std::thread> threads;
    for(int t=0;t<8;++t)threads.emplace_back([&]{for(int i=0;i<1000;++i){lifecycle.record("load","llm","shared",1,i%2==0);const auto s=lifecycle.snapshot();expect(s.recent_events.size()<=100&&s.summary.size()==1,"concurrent lifecycle snapshot");}});
    for(auto &thread:threads)thread.join();const auto s=lifecycle.snapshot();expect(s.summary[0].count==8000&&s.summary[0].failures==4000&&s.summary[0].average_seconds==1&&s.recent_events.size()==100,"concurrent lifecycle totals");
}
int available_port(){int fd=socket(AF_INET,SOCK_STREAM,0);sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);expect(!bind(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"reserve fixture port");socklen_t size=sizeof(address);getsockname(fd,reinterpret_cast<sockaddr *>(&address),&size);close(fd);return ntohs(address.sin_port);}
void environment(){for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("API_KEY","",1);setenv("DEFAULT_MODEL","fixture-model",1);setenv("N_PARALLEL","2",1);setenv("FALLBACK_SERVER","http://127.0.0.1:1",1);setenv("LLAMA_CPP_CONTEXT","128",1);setenv("LLM_BATCH_SIZE","32",1);setenv("LLM_UBATCH_SIZE","16",1);setenv("GPU_LAYERS","0",1);setenv("LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN,1);setenv("LLAMA_CPP_START_TIMEOUT","1",1);}
void integration(bool vision){
    environment();const auto directory=fs::temp_directory_path()/("ezl-model-lifecycle-"+std::to_string(getpid()));fs::create_directories(directory);const auto weights=directory/"model.gguf";{std::ofstream out(weights);out<<"GGUF fixture";}
    if(vision){const auto projector=directory/"mmproj.gguf";std::ofstream(projector)<<"fixture projector";setenv("LLAMA_CPP_MMPROJ_PATH",projector.c_str(),1);}
    const int port=available_port();setenv("LLAMA_CPP_MODEL_PATH",weights.c_str(),1);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",directory.c_str(),1);
    auto probes=ezlocalai::native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=probes.free_per_gpu=[]{return std::vector<double>{};};probes.allocator_stats=[]{return std::vector<ezlocalai::CudaAllocatorStats>{};};probes.system_ram=probes.available_ram=[]{return 128.0;};probes.now=[]{return 1000.0;};
    auto resources=std::make_shared<ezlocalai::ResourceManager>(probes);
    {
        ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=2;config.request_timeout_ms=10000;
        auto backend=std::make_unique<ezlocalai::LlamaCppInferenceBackend>("fixture-model",resources);auto *tracked=backend.get();ezlocalai::NativeServer server(config,std::move(backend),resources);
        expect(tracked->model_lifecycle_json().find("\"recent_events\":[]")!=std::string::npos,"initial lifecycle empty");
        auto request=std::async(std::launch::async,[&]{return server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"role":"user","content":"block fixture-error"}],"disable_fallback":true})");});
        struct Release { fs::path path; ~Release(){std::ofstream(path)<<"release";} } release{directory/("release-"+std::to_string(port))};
        const auto seen=directory/("seen-"+std::to_string(port));for(int i=0;i<500&&!fs::exists(seen);++i)std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if(!fs::exists(seen)){std::ofstream(directory/("release-"+std::to_string(port)))<<"release";request.get();throw std::runtime_error("managed request never entered");}
        auto snapshot=tracked->model_lifecycle_json();
        expect(snapshot.find("\"loaded_llms\":[\"fixture-model\"]")!=std::string::npos&&snapshot.find("\"active_llm\":\"fixture-model\"")!=std::string::npos,"ready model is resident before inference returns");
        expect(snapshot.find("\"gpu_layers\":0")!=std::string::npos&&snapshot.find("\"n_batch\":32")!=std::string::npos&&snapshot.find("\"n_ubatch\":16")!=std::string::npos,"loaded placement and batch runtime");
        setenv("LLAMA_CPP_CONTEXT","256",1);
        expect(tracked->model_lifecycle_json().find("\"context\":128")!=std::string::npos,"runtime shape uses captured model configuration");
        setenv("LLAMA_CPP_CONTEXT","128",1);
        expect(tracked->loaded_models_json().find(vision?"\"vision_llm\":" : "\"llm\":")!=std::string::npos,"ready text/vision model type");
        expect(tracked->loaded_models_json().find("\"active_requests\":1")!=std::string::npos,"first request activity visible");
        bool rejected=false;try{tracked->unload_llms_for_media();}catch(const std::exception &){rejected=true;}expect(rejected,"active inference prevents unload");
        const auto api=server.handle_request("GET","/v1/resources","","");expect(api.status==200&&api.body.find("\"model_lifecycle\":"+snapshot)!=std::string::npos,"resource API publishes lifecycle");
        setenv("FALLBACK_SERVER","http://127.0.0.1:1",1);setenv("FALLBACK_MEMORY_THRESHOLD","bad",1);
        auto concurrent=std::async(std::launch::async,[&]{return server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"role":"user","content":""}]})");});
        for(int i=0;i<200&&tracked->loaded_models_json().find("\"active_requests\":2")==std::string::npos;++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));
        expect(tracked->loaded_models_json().find("\"active_requests\":2")!=std::string::npos,"concurrent request uses first ready model before its initial response");
        std::ofstream(directory/("release-"+std::to_string(port)))<<"release";
        const auto failed = request.get();
        expect(failed.status==200&&failed.body.find("Fallback server error: Connection error.")!=std::string::npos&&concurrent.get().status==200,"inference fallback retains residency for concurrent local request");
        expect(tracked->loaded_models_json().find("\"active_requests\":0")!=std::string::npos,"inference activity released");
        expect(tracked->model_lifecycle_json().find("\"count\":1")!=std::string::npos,"reuse does not create extra load event");
        auto restore=tracked->unload_llms_for_media();snapshot=tracked->model_lifecycle_json();expect(snapshot.find("\"loaded_llms\":[]")!=std::string::npos&&snapshot.find("\"active_llm\":null")!=std::string::npos,"handoff clears residency");
        restore(false);expect(tracked->loaded_models_json()=="{}","disabled restoration stays unloaded");restore(true);expect(tracked->model_lifecycle_json().find("\"count\":2,\"failures\":0")!=std::string::npos,"successful restoration records load");
        restore=tracked->unload_llms_for_video();setenv("EMBEDDING_FIXTURE_FAIL_PORT",std::to_string(port).c_str(),1);restore(true);unsetenv("EMBEDDING_FIXTURE_FAIL_PORT");
        snapshot=tracked->model_lifecycle_json();expect(snapshot.find("\"loaded_llms\":[]")!=std::string::npos&&snapshot.find("\"count\":3,\"failures\":1")!=std::string::npos,"failed restoration records failure without residency");
        restore(true);snapshot=tracked->model_lifecycle_json();expect(snapshot.find("\"count\":4,\"failures\":1")!=std::string::npos&&snapshot.find("\"loaded_llms\":[\"fixture-model\"]")!=std::string::npos,"retry restoration records recovery");
    }
    std::istringstream pids(read(directory/("launch-"+std::to_string(port))));for(int pid;pids>>pid;)expect(kill(pid,0)!=0,"managed child reaped");fs::remove_all(directory);
}
void load_order(){
    environment();const auto directory=fs::temp_directory_path()/("ezl-model-order-"+std::to_string(getpid()));fs::create_directories(directory);const auto weights=directory/"model.gguf";{std::ofstream out(weights);out<<"GGUF fixture";}
    int port=0;
    for(int attempt=0;attempt<100&&!port;++attempt){const int candidate=available_port();if(candidate==65535)continue;std::vector<int> fds;bool ok=true;for(int i=0;i<2;++i){const int fd=socket(AF_INET,SOCK_STREAM,0);fds.push_back(fd);sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);address.sin_port=htons(candidate+i);if(bind(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)))ok=false;}for(int fd:fds)close(fd);if(ok)port=candidate;}
    expect(port>0,"adjacent model ports");setenv("LLAMA_CPP_MODEL_PATH",weights.c_str(),1);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",directory.c_str(),1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("one,two");auto empty=backend.unload_llms_for_media();empty(false);
        expect(backend.model_lifecycle_json().find("\"model\":\"LLM\"")!=std::string::npos,"empty handoff records reference LLM event");
        backend.chat(R"({"model":"two","messages":[]})");backend.chat(R"({"model":"one","messages":[]})");
        auto state=backend.model_lifecycle_json();expect(state.find("\"configured_llms\":[\"one\",\"two\"]")!=std::string::npos&&state.find("\"loaded_llms\":[\"two\",\"one\"]")!=std::string::npos,"loaded list preserves first-load order");
        auto restore=backend.unload_llms_for_media();state=backend.model_lifecycle_json();const auto two=state.find("\"operation\":\"unload\",\"model_type\":\"llm\",\"model\":\"two\"");const auto one=state.find("\"operation\":\"unload\",\"model_type\":\"llm\",\"model\":\"one\"");expect(two!=std::string::npos&&one!=std::string::npos&&two<one,"handoff event order follows loaded dictionary");
        restore(true);state=backend.model_lifecycle_json();expect(state.find("\"loaded_llms\":[\"two\",\"one\"]")!=std::string::npos&&state.find("\"active_llm\":\"one\"")!=std::string::npos,"restoration preserves dictionary order and active model");
    }
    for(int i=0;i<2;++i){std::istringstream pids(read(directory/("launch-"+std::to_string(port+i))));for(int pid;pids>>pid;)expect(kill(pid,0)!=0,"ordered model child reaped");}fs::remove_all(directory);
}
}
int main(int argc,char **argv){try{expect(argc==3,"reference root and fixture required");const auto n=reference_cases(argv[1],argv[2]);concurrency();integration(false);integration(true);load_order();std::cout<<"Verified "<<n<<" source lifecycle cases, concurrent recording and managed text lifetimes\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
