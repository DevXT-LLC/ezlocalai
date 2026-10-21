#include "ezlocalai/http_url.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <iomanip>
#include <openssl/sha.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <condition_variable>
#include <future>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs=std::filesystem;
namespace {
void expect(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
std::string read(const fs::path &path){std::ifstream in(path);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
std::vector<std::string> split(const std::string &value){std::vector<std::string> out;size_t pos=0;for(;;){const auto end=value.find('\t',pos);out.push_back(value.substr(pos,end==std::string::npos?end:end-pos));if(end==std::string::npos)return out;pos=end+1;}}
std::string unhex(const std::string &s){std::string out;for(size_t i=0;i<s.size();i+=2)out+=static_cast<char>(std::stoi(s.substr(i,2),nullptr,16));return out;}
std::string sha256(const std::string &value){unsigned char digest[32];SHA256(reinterpret_cast<const unsigned char *>(value.data()),value.size(),digest);std::ostringstream out;for(auto c:digest)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return out.str();}
std::vector<std::string> lines(const std::string &value){std::vector<std::string> out;std::istringstream input(value);for(std::string line;std::getline(input,line);)out.push_back(line);return out;}
std::string replace(std::string value,const std::string &from,const std::string &to){size_t pos=0;while((pos=value.find(from,pos))!=std::string::npos){value.replace(pos,from.size(),to);pos+=to.size();}return value;}
struct Step{int delay,stall;std::string wire;};
struct Scenario{
    std::mutex mutex;std::atomic<bool> probe_cookie{false};std::vector<Step> steps;size_t index=0;std::string path,first_body,trace,raw_stream;std::atomic<bool> bad_payload{false};
    std::map<std::string,std::string> urls;
    std::function<void(const std::string &)> on_request; bool probe_available=true;
    std::string expand(std::string value)const{for(const auto &url:urls)value=replace(value,"{"+url.first+"}",url.second);return value;}
};
class Remote{
public:
    Remote(Scenario &scenario,std::string tag,const std::string &host):scenario_(scenario),tag_(std::move(tag)){
        const bool v6=host=="v6";listener_=socket(v6?AF_INET6:AF_INET,SOCK_STREAM,0);expect(listener_>=0,"redirect listener");
        if(v6){sockaddr_in6 address{};address.sin6_family=AF_INET6;address.sin6_addr=in6addr_loopback;expect(!bind(listener_,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"redirect IPv6 bind");socklen_t size=sizeof(address);getsockname(listener_,reinterpret_cast<sockaddr *>(&address),&size);port_=ntohs(address.sin6_port);}
        else{sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);expect(!bind(listener_,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"redirect bind");socklen_t size=sizeof(address);getsockname(listener_,reinterpret_cast<sockaddr *>(&address),&size);port_=ntohs(address.sin_port);}
        url="http://"+std::string(v6?"[::1]":host=="dns"?"localhost":"127.0.0.1")+":"+std::to_string(port_);
        expect(!listen(listener_,16),"redirect listen");thread_=std::thread([&]{while(!stopped_){const int peer=accept(listener_,nullptr,nullptr);if(peer<0)continue;workers_.emplace_back([&,peer]{serve(peer);shutdown(peer,SHUT_RDWR);close(peer);});}});
    }
    ~Remote(){stopped_=true;shutdown(listener_,SHUT_RDWR);thread_.join();for(auto &worker:workers_)worker.join();close(listener_);}
    std::string url;
private:
    Scenario &scenario_;std::string tag_;int listener_=-1,port_=0;std::atomic<bool> stopped_{false};std::thread thread_;std::vector<std::thread> workers_;
    void pause(int milliseconds){for(int i=0;i<milliseconds&&!stopped_;i+=2)std::this_thread::sleep_for(std::chrono::milliseconds(std::min(2,milliseconds-i)));}
    void send(int peer,const std::string &value){for(size_t pos=0;pos<value.size();){const auto n=::send(peer,value.data()+pos,value.size()-pos,MSG_NOSIGNAL);if(n<=0)return;pos+=n;}}
    void serve(int peer){
        timeval timeout{2,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));std::string request;char buffer[4096];size_t end=0,length=0;std::map<std::string,std::string> headers;
        for(;;){const auto n=recv(peer,buffer,sizeof(buffer),0);if(n<=0)return;request.append(buffer,n);end=request.find("\r\n\r\n");if(end==std::string::npos)continue;
            for(auto line:lines(request.substr(0,end))){if(!line.empty()&&line.back()=='\r')line.pop_back();const auto colon=line.find(':');if(colon==std::string::npos)continue;auto key=line.substr(0,colon);for(auto &c:key)c=std::tolower(static_cast<unsigned char>(c));const auto start=line.find_first_not_of(" \t",colon+1);headers[key]=start==std::string::npos?"":line.substr(start);}
            length=headers.count("content-length")?std::stoul(headers["content-length"]):0;if(request.size()>=end+4+length)break;}
        std::istringstream first(request);std::string method,path;first>>method>>path;
        if(scenario_.on_request)scenario_.on_request(path=="/v1/resources"?"remote:probe":"remote:post");
        Step step;{
            std::lock_guard<std::mutex> lock(scenario_.mutex);
            const bool target=(scenario_.index>0&&scenario_.index<scenario_.steps.size())||path==scenario_.path||(path!="/v1/resources"&&path!="/v1/models");
            if(target){
                const auto body=request.substr(end+4,length);if(!body.empty()&&scenario_.first_body.empty())scenario_.first_body=body;
                auto type=headers["content-type"];if(type.find("multipart/form-data;")==0)type="multipart/form-data";
                if(!scenario_.trace.empty())scenario_.trace+='\n';scenario_.trace+=tag_+'\t'+method+'\t'+path+'\t'+headers["authorization"]+'\t'+headers["cookie"]+'\t'+(body.empty()?"empty":"payload");
                if(!body.empty()&&(body.find("\"max_tokens\":8192")==std::string::npos||body.find("\"stream\":"+scenario_.raw_stream)==std::string::npos))scenario_.bad_payload=true;
                if(scenario_.index<scenario_.steps.size())step=scenario_.steps[scenario_.index++];else step={0,0,"HTTP/1.1 500 Bad\r\nContent-Length: 0\r\n\r\n"};
                step.wire=scenario_.expand(step.wire);
            }else{if(!headers["cookie"].empty())scenario_.probe_cookie=true;const auto body=path=="/v1/resources"?std::string("{\"free_vram_gb\":100}"):std::string("{\"data\":[]}");step={0,0,std::string(scenario_.probe_available?"HTTP/1.1 200 OK":"HTTP/1.1 503 Bad")+"\r\nContent-Type: application/json\r\nSet-Cookie: probe=isolated; Path=/\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body};}
        }
        pause(step.delay);send(peer,step.wire);pause(step.stall);
    }
};
struct Gate {
    std::mutex mutex; std::condition_variable cv; bool entered=false, released=false;
    void hold(){std::unique_lock<std::mutex> lock(mutex);entered=true;cv.notify_all();cv.wait(lock,[&]{return released;});}
    void wait(){std::unique_lock<std::mutex> lock(mutex);expect(cv.wait_for(lock,std::chrono::seconds(2),[&]{return entered;}),"local queue holder entered");}
    void release(){std::lock_guard<std::mutex> lock(mutex);released=true;cv.notify_all();}
};
class Backend final:public ezlocalai::InferenceBackend {
public:
    Gate gate;std::atomic<unsigned> targets{0};bool streaming=false;std::atomic<bool> control_leaked{false};std::string loaded="{}";std::function<void()> on_call;
    std::string loaded_models_json()const override{return loaded;}
    std::string chat(const std::string &body) override {
        if(body.find("_fixture_hold")!=std::string::npos){gate.hold();return R"({"marker":"holder"})";}
        ++targets;if(body.find("disable_fallback")!=std::string::npos)control_leaked=true;if(on_call)on_call();return streaming?"data: {\"marker\": \"local\"}\n\ndata: [DONE]\n\n":R"({"marker":"local"})";
    }
    std::string completion(const std::string &body)override{return chat(body);}
    std::string embedding(const std::string &)override{return "{}";}
};
size_t api_cases(const fs::path &root,const fs::path &path) {
    size_t count=0;std::istringstream input(read(path));
    for(std::string line;std::getline(input,line);){
        if(line.empty())continue;
        if(line[0]=='#'){const auto marker=line.find(" sha256=");if(marker!=std::string::npos)expect(sha256(read(root/line.substr(2,marker-2)))==line.substr(marker+8),"LLM fallback reference source changed");continue;}
        ++count;const auto c=split(line);Scenario scenario;scenario.path=c[2]=="chat"?"/v1/chat/completions":"/v1/completions";scenario.raw_stream=unhex(c[3]);
        for(const auto &encoded:lines(unhex(c[5]))){const auto step=split(encoded);scenario.steps.push_back({std::stoi(step[0]),std::stoi(step[1]),unhex(step[2])});}
        std::mutex posted_mutex;std::condition_variable posted_cv;bool posted=false,response_complete=false;
        scenario.on_request=[&](const std::string &event){if(event=="remote:post"){std::lock_guard<std::mutex> lock(posted_mutex);posted=true;posted_cv.notify_all();}};
        Remote a(scenario,"A","dns"),b(scenario,"B","dns");scenario.urls={{"A",a.url},{"B",b.url}};
        for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());
        setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("API_KEY","",1);setenv("DEFAULT_MODEL","test-model",1);setenv("FALLBACK_SERVER",a.url.c_str(),1);setenv("FALLBACK_API_KEY","secret",1);setenv("QUEUE_WAIT_TIMEOUT","0.001",1);setenv("REQUEST_TIMEOUT",c[4].c_str(),1);
        auto probes=ezlocalai::native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.free_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 128.0;};probes.available_ram=[]{return 128.0;};
        auto backend=std::make_unique<Backend>();auto *tracked=backend.get();tracked->streaming=scenario.raw_stream!="false"&&scenario.raw_stream!="null"&&scenario.raw_stream!="0"&&scenario.raw_stream!="0.0";
        ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=1;config.request_timeout_ms=2000;
        auto resources=std::make_shared<ezlocalai::ResourceManager>(probes);
        resources->register_model(ezlocalai::ModelType::Llm,"queue holder","cpu",0);
        ezlocalai::NativeServer server(config,std::move(backend),resources);
        auto holding=std::async(std::launch::async,[&]{return server.handle_request("POST","/v1/chat/completions","",R"({"messages":[],"disable_fallback":true,"_fixture_hold":true})");});
        tracked->gate.wait();
        // Successful fallback must finish/cancel its queued request before the
        // holder exits. Recovery cases need the local queue after the remote
        // attempt starts. Neither boundary depends on a fixed scheduling delay.
        std::thread release([&]{std::unique_lock<std::mutex> lock(posted_mutex);posted_cv.wait_for(lock,std::chrono::seconds(5),[&]{return response_complete||(c[9]!="1"&&posted);});lock.unlock();tracked->gate.release();});
        const auto started=std::chrono::steady_clock::now();
        const auto body=std::string(c[2]=="chat"?"{\"messages\":[],\"stream\":" : "{\"prompt\":\"test\",\"stream\":")+scenario.raw_stream+"}";
        const auto response=server.handle_request("POST",scenario.path,"",body);
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        {std::lock_guard<std::mutex> lock(posted_mutex);response_complete=true;posted_cv.notify_all();}
        release.join();holding.get();
        const auto label="source LLM fallback "+c[0]+" "+c[1]+" "+c[2]+" stream="+scenario.raw_stream;
        expect(response.status==std::stoi(c[6])&&response.content_type==unhex(c[7])&&response.body==unhex(c[8]),label+" response: "+std::to_string(response.status)+" "+response.content_type+" "+response.body+" expected "+unhex(c[8]));
        expect(tracked->targets==(c[9]=="1"?0U:1U),label+" local recovery");
        std::string trace;{std::lock_guard<std::mutex> lock(scenario.mutex);trace=scenario.trace;}
        expect(trace==unhex(c[10]),label+" trace:\n"+trace+"\nexpected:\n"+unhex(c[10]));
        expect(!scenario.bad_payload&&!scenario.probe_cookie,label+" request payload and per-call cookies");
        if(c[9]=="1"&&(c[1].find("stalled-")==0||c[1].find("redirect-")==0||c[1].find("chunked-")==0))expect(elapsed<.07,label+" releases successful response without consuming the stream");
    }
    expect(count==178,"complete LLM fallback API fixtures");return count;
}
size_t precheck_cases(const fs::path &root,const fs::path &path) {
    size_t count=0;std::istringstream input(read(path));
    for(std::string line;std::getline(input,line);){
        if(line.empty())continue;
        if(line[0]=='#'){const auto marker=line.find(" sha256=");if(marker!=std::string::npos)expect(sha256(read(root/line.substr(2,marker-2)))==line.substr(marker+8),"LLM precheck reference source changed");continue;}
        ++count;const auto c=split(line);expect(c.size()==18,"precheck fixture columns");
        Scenario scenario;scenario.path=c[1]=="chat"?"/v1/chat/completions":"/v1/completions";scenario.raw_stream=unhex(c[2]);scenario.probe_available=c[13]!="unavailable";
        const std::string remote_body=c[13]=="bad_json"?"not JSON":R"({"marker":"remote"})";
        scenario.steps.push_back({0,0,std::string(c[13]=="error"?"HTTP/1.1 503 Bad":"HTTP/1.1 200 OK")+"\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(remote_body.size())+"\r\n\r\n"+remote_body});
        std::mutex trace_mutex;std::string trace;
        const auto event=[&](const std::string &name){std::lock_guard<std::mutex> lock(trace_mutex);if(!trace.empty())trace+=',';trace+=name;};
        scenario.on_request=event;Remote remote(scenario,"A","dns");
        for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());
        setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("API_KEY","",1);setenv("DEFAULT_MODEL","test-model",1);setenv("FALLBACK_SERVER",c[5]=="1"?remote.url.c_str():"",1);setenv("FALLBACK_API_KEY","secret",1);setenv("QUEUE_WAIT_TIMEOUT","30",1);setenv("REQUEST_TIMEOUT","2",1);
        if(c[9]!="N")setenv("FALLBACK_MEMORY_THRESHOLD",unhex(c[9].substr(1)).c_str(),1);
        std::vector<std::vector<double>> readings;std::istringstream encoded(c[7]);
        for(std::string row;std::getline(encoded,row,'/');){std::vector<double> values;if(row!="-"){std::istringstream fields(row);for(std::string n;std::getline(fields,n,',');)values.push_back(std::stod(n));}readings.push_back(std::move(values));}
        size_t reading=0;auto probes=ezlocalai::native_resource_probes();
        probes.gpu_count=[&]{return int(readings.front().size());};probes.total_per_gpu=[&]{return std::vector<double>(readings.front().size(),24);};
        probes.free_per_gpu=[&]{event("free");if(c[12]=="free"||(c[12]=="admission"&&reading==1))throw std::runtime_error("memory fixture failed");return readings[std::min(reading++,readings.size()-1)];};
        probes.system_ram=[]{return 128.0;};probes.available_ram=[&]{event("ram");if(c[12]=="ram")throw std::runtime_error("RAM fixture failed");return std::stod(c[8]);};
        probes.primary_gpu=[&]{event("primary");return 0;};probes.secondary_gpu=[&]()->std::optional<int>{event("secondary");return readings.front().size()>1?std::optional<int>(1):std::nullopt;};
        probes.fallback_server=[&]{event("fallback");return c[6]=="1"?remote.url:std::string();};probes.now=[]{return 1.0;};
        auto resources=std::make_shared<ezlocalai::ResourceManager>(probes);
        if(c[4]=="llm"||c[4]=="vision_llm")resources->register_model(c[4]=="llm"?ezlocalai::ModelType::Llm:ezlocalai::ModelType::VisionLlm,"model","cuda",8);
        const auto idle=std::stod(c[10]);if(idle){resources->register_model(ezlocalai::ModelType::Embedding,"embedding","cuda",idle);resources->mark_model_in_use(ezlocalai::ModelType::Embedding,c[11]=="1");}
        auto backend=std::make_unique<Backend>();auto *tracked=backend.get();tracked->streaming=scenario.raw_stream!="false"&&scenario.raw_stream!="null"&&scenario.raw_stream!="0"&&scenario.raw_stream!="0.0";
        if(c[4]=="backend_llm")tracked->loaded=R"({"llm":{}})";if(c[4]=="backend_vision")tracked->loaded=R"({"vision_llm":{}})";
        tracked->on_call=[&]{event("local");};
        ezl_env_config config{};ezl_load_config(&config);config.max_concurrent_requests=1;config.request_timeout_ms=2000;
        ezlocalai::NativeServer server(config,std::move(backend),resources);
        auto body=std::string(c[1]=="chat"?"{\"messages\":[],\"stream\":" : "{\"prompt\":\"test\",\"stream\":")+scenario.raw_stream;
        if(c[3]!="N")body+=",\"disable_fallback\":"+unhex(c[3]);body+='}';
        const auto response=server.handle_request("POST",scenario.path,"",body);
        const auto label="source LLM precheck "+c[0]+" "+c[1]+" resident="+c[4]+" mode="+c[13];
        expect(response.status==std::stoi(c[14])&&response.content_type==unhex(c[15])&&response.body==unhex(c[16]),label+" response: "+std::to_string(response.status)+" "+response.content_type+" "+response.body+" expected "+unhex(c[16]));
        expect(trace==unhex(c[17]),label+" events: "+trace+" expected "+unhex(c[17]));
        expect(!tracked->control_leaked&&!scenario.bad_payload&&!scenario.probe_cookie&&scenario.first_body.find("disable_fallback")==std::string::npos,label+" forwarded payload controls");
    }
    expect(count==816,"complete LLM precheck fixtures");return count;
}
void managed_residency_and_coercion() {
    Scenario scenario;scenario.path="/v1/chat/completions";scenario.raw_stream="false";
    const auto wire=[](const std::string &body){return "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body;};
    scenario.steps.push_back({0,0,wire(R"({"marker":"loaded"})")});
    const std::string chunks="data: {\"choices\":[{\"delta\":{\"content\":\"one\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"two\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    for(int i=0;i<6;++i)scenario.steps.push_back({0,0,wire(chunks)});
    std::atomic<unsigned> probes{0};scenario.on_request=[&](const std::string &name){if(name=="remote:probe")++probes;};Remote remote(scenario,"A","dns");
    for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());
    setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("API_KEY","",1);setenv("DEFAULT_MODEL","test-model",1);setenv("LLAMA_CPP_SERVER_URL",remote.url.c_str(),1);setenv("FALLBACK_SERVER",remote.url.c_str(),1);setenv("FALLBACK_MEMORY_THRESHOLD","bad",1);
    auto p=ezlocalai::native_resource_probes();p.gpu_count=[]{return 0;};p.total_per_gpu=[]{return std::vector<double>{};};p.system_ram=[]{return 128.0;};p.available_ram=[]{throw std::runtime_error("resident LLM must skip RAM probe");return 0.0;};
    ezl_env_config config{};ezl_load_config(&config);config.request_timeout_ms=2000;
    ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model),std::make_shared<ezlocalai::ResourceManager>(p));
    auto response=server.handle_request("POST",scenario.path,"",R"({"messages":[{"role":"user","content":""}],"stream":false,"disable_fallback":true})");
    expect(response.status==200&&response.body==R"({"marker":"loaded","model":"test-model"})","managed text model initial local request");
    for(const std::string value:{"false","off","0","no","f","n"}){
        scenario.raw_stream="true";
        response=server.handle_request("POST",scenario.path,"","{\"messages\":[{\"role\":\"user\",\"content\":\"\"}],\"stream\":\""+value+"\"}");
        expect(response.status==200&&response.content_type=="application/json"&&response.body==R"([{"choices":[{"delta":{"content":"one"}}]},{"choices":[{"delta":{"content":"two"}}]},{"choices":[{"delta":{},"finish_reason":"stop"}]}])","managed raw/validated stream coercion "+value);
    }
    expect(probes==0&&!scenario.bad_payload,"resident engine skips fallback and receives the chat generator's stream=true");
    unsetenv("LLAMA_CPP_SERVER_URL");
}
}
int main(int argc,char **argv){try{expect(argc==3,"reference root and fixtures required");const auto n=api_cases(argv[1],fs::path(argv[2])/"llm_fallback_cases.tsv");const auto m=precheck_cases(argv[1],fs::path(argv[2])/"llm_precheck_cases.tsv");managed_residency_and_coercion();std::cout<<"Verified "<<n<<" source LLM fallback API cases and "<<m<<" precheck cases through the native queue\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
