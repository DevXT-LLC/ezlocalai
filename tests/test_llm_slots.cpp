#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
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
std::string hex(const std::string &s){std::ostringstream out;for(unsigned char c:s)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return out.str();}
std::string sha256(const std::string &value){unsigned char digest[32];SHA256(reinterpret_cast<const unsigned char *>(value.data()),value.size(),digest);return hex(std::string(reinterpret_cast<char *>(digest),32));}
std::optional<std::string> optional(const std::string &s){return s=="N"?std::nullopt:std::optional<std::string>(unhex(s.substr(1)));}
std::string encoded(const std::optional<std::string> &s){return s?"S"+hex(*s):"N";}
std::map<std::string,int64_t> counts(const std::string &s){std::map<std::string,int64_t> out;if(!s.empty())for(const auto &row:split(unhex(s),'\n')){const auto c=split(row);out[unhex(c[0])]=std::stoll(c[1]);}return out;}
size_t reference_cases(const fs::path &root,const fs::path &fixture){
    size_t count=0;std::istringstream input(read(fixture));
    for(std::string line;std::getline(input,line);){
        if(line.empty())continue;if(line[0]=='#'){const auto marker=line.find(" sha256=");if(marker!=std::string::npos)expect(sha256(read(root/line.substr(2,marker-2)))==line.substr(marker+8),"slot source changed");continue;}
        ++count;const auto c=split(line);std::vector<ezlocalai::LlmReplica> replicas;
        if(!c[1].empty())for(const auto &row:split(unhex(c[1]),'\n')){const auto r=split(row);replicas.push_back({unhex(r[0]),unhex(r[1]),std::stoll(r[2]),std::stoll(r[3]),r[4]=="N"?std::nullopt:std::optional<int64_t>(std::stoll(r[4])),r[5]=="1",r[6]=="1"});}
        ezlocalai::LlmSlotState state{std::stoll(c[2]),counts(c[3]),counts(c[4])};const auto expected=split(unhex(c[6]),'\n');size_t i=0;
        for(const auto &row:split(unhex(c[5]),'\n')){const auto op=split(row);std::string value;
            if(op[0]=="q")value=encoded(ezlocalai::resolve_llm_replica(replicas,unhex(op[1])))+","+encoded(ezlocalai::resolve_llm_slot_model(replicas,unhex(op[1])));
            else if(op[0]=="c")value=std::to_string(ezlocalai::llm_model_capacity(replicas,optional(op[1]),op[2]=="1"));
            else if(op[0]=="a"){const auto result=ezlocalai::try_acquire_llm_slot(replicas,state,optional(op[1]),optional(op[2]),op[3]=="1",op[4]=="1");value=result.acquired?"ok:"+encoded(result.replica):"wait";}
            else{ezlocalai::release_llm_slot(state,optional(op[1]),optional(op[2]));value="released";}
            const auto e=split(expected.at(i++));expect(value==unhex(e[0])&&state.total==std::stoll(e[1])&&state.models==counts(e[2])&&state.replicas==counts(e[3]),"source slot case "+c[0]+" operation "+std::to_string(i)+" got "+value+" expected "+unhex(e[0]));
        }
    }
    expect(count==1084,"all slot reference cases");return count;
}
void environment(){for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("API_KEY","",1);setenv("DEFAULT_MODEL","org/model,org/model,other",1);setenv("N_PARALLEL","1",1);}
struct Recording {std::mutex mutex;std::condition_variable cv;std::vector<std::string> selected;std::map<std::string,bool> released;bool all=false;
    void call(const std::string &replica,const std::string &body){std::unique_lock<std::mutex> lock(mutex);selected.push_back(replica);cv.notify_all();if(body.find("fixture-error")!=std::string::npos)throw std::runtime_error("fixture inference failed");const auto pos=body.find("\"hold\":\"");if(pos!=std::string::npos){const auto start=pos+8;const auto token=body.substr(start,body.find('"',start)-start);if(token.size()==1)expect(cv.wait_for(lock,std::chrono::seconds(3),[&]{return all||released[token];}),"held fixture request released");}}
    void wait(size_t n){std::unique_lock<std::mutex> lock(mutex);expect(cv.wait_for(lock,std::chrono::seconds(3),[&]{return selected.size()>=n;}),"replica call entered");}
    void release(const std::string &token){std::lock_guard<std::mutex> lock(mutex);released[token]=true;cv.notify_all();}
    std::vector<std::string> calls(){std::lock_guard<std::mutex> lock(mutex);return selected;}
    void finish(){std::lock_guard<std::mutex> lock(mutex);all=true;cv.notify_all();}
};
class Backend final:public ezlocalai::InferenceBackend {
public:std::shared_ptr<Recording> state;explicit Backend(std::shared_ptr<Recording> s):state(std::move(s)){}
    std::string chat(const std::string &)override{throw std::runtime_error("replica was not provided");}std::string completion(const std::string &s)override{return chat(s);}std::string embedding(const std::string &)override{return "{}";}
    std::string chat_for_replica(const std::string &body,const std::string &replica)override{state->call(replica,body);return "{}";}
    std::string completion_for_replica(const std::string &body,const std::string &replica)override{return chat_for_replica(body,replica);}
};
void scheduling(){
    environment();auto state=std::make_shared<Recording>();ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=8;config.request_timeout_ms=5000;
    ezlocalai::NativeServer server(config,std::make_unique<Backend>(state));
    const auto request=[&](const std::string &model,const std::string &token,const std::string &path="/v1/chat/completions"){return server.handle_request("POST",path,"","{\"messages\":[],\"prompt\":\"test\",\"model\":\""+model+"\",\"hold\":\""+token+"\"}");};
    auto a=std::async(std::launch::async,[&]{return request("MODEL","a");});state->wait(1);
    auto b=std::async(std::launch::async,[&]{return request("org/model","b");});state->wait(2);
    auto c=std::async(std::launch::async,[&]{return request("model","c");});
    struct Finish{std::shared_ptr<Recording>s;~Finish(){s->finish();}} finish{state};
    expect(c.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout&&state->calls()==std::vector<std::string>{"org/model","org/model#2"},"least-busy replicas and full-model waiting");
    const auto resources=server.handle_request("GET","/v1/resources","","");expect(resources.body.find("\"org/model\":{\"capacity\":2,\"in_flight\":2")!=std::string::npos,"waiting request does not consume model slots");
    expect(request("other","independent").status==200&&state->calls().back()=="other","another model has independent capacity");
    state->release("a");expect(a.get().status==200,"first request finishes");state->wait(4);expect(state->calls().back()=="org/model","released replica reused");state->release("b");state->release("c");b.get();c.get();
    auto d=std::async(std::launch::async,[&]{return request("MODEL#2","d","/v1/completions");});state->wait(5);
    auto e=std::async(std::launch::async,[&]{return request("org/model#2","e");});expect(e.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout,"explicit busy replica waits even if sibling idle");
    expect(request("org/model","fast").status==200&&state->calls().back()=="org/model","public request uses idle sibling");
    state->release("d");d.get();state->wait(7);expect(state->calls().back()=="org/model#2","explicit waiter keeps requested replica");state->release("e");e.get();
    expect(request("org/model#2","fixture-error").status==500,"failed backend request returned");expect(request("org/model#2","after-error").status==200,"failed request releases its replica");
}
int available_port(){const int fd=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);expect(!bind(fd,reinterpret_cast<sockaddr *>(&a),sizeof(a)),"fixture port");socklen_t n=sizeof(a);getsockname(fd,reinterpret_cast<sockaddr *>(&a),&n);close(fd);return ntohs(a.sin_port);}
int adjacent_ports(int count=3){
    for(int attempt=0;attempt<100;++attempt){const int candidate=available_port();if(candidate>65536-count)continue;std::vector<int> sockets;bool free=true;for(int i=0;i<count;++i){const int fd=socket(AF_INET,SOCK_STREAM,0);sockets.push_back(fd);sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);a.sin_port=htons(candidate+i);if(bind(fd,reinterpret_cast<sockaddr *>(&a),sizeof(a)))free=false;}for(int fd:sockets)close(fd);if(free)return candidate;}
    throw std::runtime_error("adjacent fixture ports unavailable");
}
void automatic_scheduling(){
    environment();setenv("DEFAULT_MODEL","org/model-mtp",1);setenv("N_PARALLEL","3",1);
    auto state=std::make_shared<Recording>();ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=8;config.request_timeout_ms=5000;
    ezlocalai::NativeServer server(config,std::make_unique<Backend>(state));
    const auto request=[&](const std::string &token){return server.handle_request("POST","/v1/chat/completions","","{\"messages\":[],\"model\":\"model-mtp\",\"hold\":\""+token+"\"}");};
    auto a=std::async(std::launch::async,[&]{return request("a");});state->wait(1);
    auto b=std::async(std::launch::async,[&]{return request("b");});state->wait(2);
    auto c=std::async(std::launch::async,[&]{return request("c");});state->wait(3);
    auto d=std::async(std::launch::async,[&]{return request("d");});
    struct Finish{std::shared_ptr<Recording>s;~Finish(){s->finish();}} finish{state};
    expect(d.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout&&state->calls()==std::vector<std::string>{"org/model-mtp","org/model-mtp#2","org/model-mtp#3"},"automatic replicas each have one independent slot");
    state->release("b");b.get();state->wait(4);expect(state->calls().back()=="org/model-mtp#2","automatic replica reused after release");state->finish();expect(a.get().status==200&&c.get().status==200&&d.get().status==200,"automatic requests finish");
}
void automatic_managed_selection(){
    environment();setenv("DEFAULT_MODEL","org/model-mtp,org/other,org/literal#2",1);setenv("N_PARALLEL","3,2,1",1);setenv("MAIN_GPU","2,3,4",1);setenv("LLAMA_CPP_CONTEXT","65536,32768,1024",1);setenv("GPU_LAYERS","0",1);setenv("LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN,1);setenv("LLAMA_CPP_START_TIMEOUT","2",1);
    const auto dir=fs::temp_directory_path()/("ezl-auto-slots-"+std::to_string(getpid()));fs::create_directories(dir);std::string paths;
    for(const auto &name:{"first.gguf","second.gguf","literal.gguf"}){const auto path=dir/name;std::ofstream(path)<<"GGUF fixture";if(!paths.empty())paths+=',';paths+=path.string();}
    setenv("LLAMA_CPP_MODEL_PATHS",paths.c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",dir.c_str(),1);const int port=adjacent_ports(5);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);
    {
        ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=8;config.request_timeout_ms=5000;ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
        const auto result=server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"role":"user","content":""}],"model":"model-mtp#3"})");expect(result.status==200,"automatic managed replica responds");
        expect(fs::exists(dir/("launch-"+std::to_string(port+4)))&&!fs::exists(dir/("launch-"+std::to_string(port))),"automatic instances append after original entries");
        const auto args=read(dir/("args-"+std::to_string(port+4)));expect(args.find("-m\n"+(dir/"first.gguf").string()+"\n")!=std::string::npos&&args.find("-np\n1\n")!=std::string::npos&&args.find("-c\n65536\n")!=std::string::npos&&args.find("--main-gpu\n2\n")!=std::string::npos,"automatic replica inherits original model path, parallelism, context and GPU");
        const auto literal=server.handle_request("POST","/v1/completions","",R"({"prompt":"test","model":"literal#2"})");expect(literal.status==200&&literal.body.find("\"model\":\"org/literal#2\"")!=std::string::npos&&read(dir/("seen-"+std::to_string(port+2))).find("\"model\"")==std::string::npos,"numeric source suffix selects the engine and names the response");
        const auto resources=server.handle_request("GET","/v1/resources","","");expect(resources.body.find("\"org/model-mtp\":{\"capacity\":3,")!=std::string::npos&&resources.body.find("\"org/other\":{\"capacity\":2,")!=std::string::npos,"expanded resources advertise aggregate capacity");
        const auto models=server.handle_request("GET","/v1/models","","");expect(models.body.find("org/literal#2")!=std::string::npos&&models.body.find("model-mtp#3")==std::string::npos,"model list exposes original public names");
    }
    setenv("DEFAULT_MODEL","a,a#2,a",1);setenv("N_PARALLEL","1",1);
    {
        ezl_env_config config{};ezl_load_config(&config);ezlocalai::NativeServer server(config,std::make_unique<Backend>(std::make_shared<Recording>()));
        const auto models=server.handle_request("GET","/v1/models","","");expect(models.body==R"({"object":"list","data":[{"id":"a","object":"model","owned_by":"ezlocalai"}]})","model listing resolves colliding IDs before deduplicating");
    }
    for(int i=0;i<5;++i){std::istringstream pids(read(dir/("launch-"+std::to_string(port+i))));for(int pid;pids>>pid;)expect(kill(pid,0)!=0,"automatic child reaped");}fs::remove_all(dir);
}

void managed_selection(){
    environment();setenv("N_PARALLEL","0",1);setenv("LLAMA_CPP_CONTEXT","65536",1);setenv("GPU_LAYERS","0",1);setenv("LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN,1);setenv("LLAMA_CPP_START_TIMEOUT","2",1);
    const auto dir=fs::temp_directory_path()/("ezl-slots-"+std::to_string(getpid()));fs::create_directories(dir);const auto model=dir/"model.gguf";std::ofstream(model)<<"GGUF fixture";setenv("LLAMA_CPP_MODEL_PATH",model.c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",dir.c_str(),1);const int port=adjacent_ports();setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);
    {
        ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=4;config.request_timeout_ms=5000;
        ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
        setenv("N_PARALLEL","1",1);
        const auto result=server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"role":"user","content":""}],"model":"MODEL#2"})");expect(result.status==200,"explicit managed replica request");
        expect(!fs::exists(dir/("launch-"+std::to_string(port)))&&fs::exists(dir/("launch-"+std::to_string(port+1))),"only requested replica started");
        const auto body=read(dir/("seen-"+std::to_string(port+1)));expect(body.find("\"model\"")==std::string::npos&&body.find("MODEL#2")==std::string::npos&&result.body.find("\"model\":\"org/model\"")!=std::string::npos,"selected engine receives no model control and response names its source");
        expect(read(dir/("args-"+std::to_string(port+1))).find("-np\n2\n")!=std::string::npos,"automatic parallelism reaches engine");
        const auto resources=server.handle_request("GET","/v1/resources","","");expect(resources.body.find("\"org/model\":{\"capacity\":4,")!=std::string::npos,"model capacity uses captured automatic parallelism");
        const auto second=server.handle_request("POST","/v1/completions","",R"({"prompt":"test","model":"unknown"})");expect(second.status==200&&fs::exists(dir/("launch-"+std::to_string(port))),"unknown model uses first public model");
    }
    for(int i=0;i<3;++i){std::istringstream pids(read(dir/("launch-"+std::to_string(port+i))));for(int pid;pids>>pid;)expect(kill(pid,0)!=0,"replica process reaped");}fs::remove_all(dir);
}
}
int main(int argc,char **argv){try{expect(argc==3,"reference root and fixture required");const auto count=reference_cases(argv[1],argv[2]);scheduling();managed_selection();automatic_scheduling();automatic_managed_selection();std::cout<<"Verified "<<count<<" source replica/slot cases and native concurrent/managed routing\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
