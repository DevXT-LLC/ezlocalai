#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <openssl/sha.h>
#include <sstream>
#include <thread>
#include <unistd.h>
namespace fs=std::filesystem;
namespace {
void expect(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
std::string read(const fs::path &path){std::ifstream in(path);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
std::string hash(const std::string &value){unsigned char digest[32];SHA256(reinterpret_cast<const unsigned char *>(value.data()),value.size(),digest);std::ostringstream out;for(auto c:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return out.str();}
std::vector<std::string> split(const std::string &s){std::vector<std::string> out;for(size_t at=0;;){const auto end=s.find('\t',at);out.push_back(s.substr(at,end==std::string::npos?end:end-at));if(end==std::string::npos)return out;at=end+1;}}
std::string unhex(const std::string &s){std::string out;for(size_t i=0;i<s.size();i+=2)out+=static_cast<char>(std::stoi(s.substr(i,2),nullptr,16));return out;}
std::string quote(const std::string &s){return '"'+ezlocalai::json_escape(s)+'"';}
class Remote {
public:
    std::string url;
    explicit Remote(const std::vector<std::string> &c):c_(c){
        fd_=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        expect(!bind(fd_,reinterpret_cast<sockaddr *>(&a),sizeof(a)),"fallback fixture bind");socklen_t size=sizeof(a);getsockname(fd_,reinterpret_cast<sockaddr *>(&a),&size);
        url="http://127.0.0.1:"+std::to_string(ntohs(a.sin_port))+unhex(c_[9]);expect(!listen(fd_,8),"fallback fixture listen");
        thread_=std::thread([&]{while(!stopped_){int peer=accept(fd_,nullptr,nullptr);if(peer<0)continue;serve(peer);shutdown(peer,SHUT_RDWR);close(peer);}});
    }
    ~Remote(){stopped_=true;shutdown(fd_,SHUT_RDWR);thread_.join();close(fd_);}
    std::string trace(){std::lock_guard<std::mutex> lock(mutex_);return '['+trace_+']';}
private:
    const std::vector<std::string> &c_;int fd_;std::atomic<bool> stopped_{false};std::thread thread_;std::mutex mutex_;std::string trace_;
    void serve(int peer){
        timeval timeout{2,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));std::string request;char buffer[4096];size_t end=0,length=0;std::string auth;
        for(;;){auto n=recv(peer,buffer,sizeof(buffer),0);if(n<=0)return;request.append(buffer,n);end=request.find("\r\n\r\n");if(end==std::string::npos)continue;
            std::istringstream headers(request.substr(0,end));for(std::string line;std::getline(headers,line);){if(!line.empty()&&line.back()=='\r')line.pop_back();auto colon=line.find(':');if(colon==std::string::npos)continue;auto key=line.substr(0,colon);for(auto &ch:key)ch=std::tolower(static_cast<unsigned char>(ch));auto value=line.substr(colon+1);while(!value.empty()&&value.front()==' ')value.erase(value.begin());if(key=="content-length")length=std::stoul(value);if(key=="authorization")auth=value;}
            if(request.size()>=end+4+length)break;
        }
        std::istringstream first(request);std::string method,path;first>>method>>path;const auto body=request.substr(end+4,length);
        {std::lock_guard<std::mutex> lock(mutex_);if(!trace_.empty())trace_+=',';trace_+='['+quote(method)+','+quote(path)+','+quote(auth)+','+(body.empty()?"null":body)+']';}
        const bool probe=path==unhex(c_[9])+"/v1/resources",ezlocal=path==unhex(c_[9])+"/v1/chat/completions";
        const int status=probe?(c_[3]=="1"?200:503):std::stoi(c_[ezlocal?4:6]);
        const auto response_body=probe?std::string("{\"free_vram_gb\":100}"):unhex(c_[ezlocal?5:7]);
        const std::string wire="HTTP/1.1 "+std::to_string(status)+" Fixture\r\nContent-Type: application/json\r\nretry-after-ms: 1\r\nContent-Length: "+std::to_string(response_body.size())+"\r\nConnection: close\r\n\r\n"+response_body;
        for(size_t at=0;at<wire.size();){const auto n=send(peer,wire.data()+at,wire.size()-at,MSG_NOSIGNAL);if(n<=0)return;at+=n;}
    }
};
class Backend final:public ezlocalai::InferenceBackend {
public:
    unsigned calls=0;
    std::string chat(const std::string &)override{return "{}";}
    std::string completion(const std::string &)override{return "{}";}
    std::string embedding(const std::string &)override{return "{}";}
    ezlocalai::InferenceResult text_result(const std::string &body,const std::string &,bool)override{
        ++calls;expect(body.find("disable_fallback")==std::string::npos,"control removed before local inference");
        return {R"({"error":{"message":"local inference failed"}})",{"Exception","local inference failed"}};
    }
};
void environment(){for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("DEFAULT_MODEL","default-model",1);setenv("FALLBACK_API_KEY","secret",1);setenv("REQUEST_TIMEOUT","3",1);}
}
int main(int argc,char **argv){try{
    expect(argc==3,"reference and fixture arguments");std::istringstream input(read(argv[2]));size_t count=0;
    for(std::string line;std::getline(input,line);){if(line.empty())continue;if(line.front()=='#'){const auto at=line.find(" sha256=");if(at!=std::string::npos)expect(hash(read(fs::path(argv[1])/line.substr(2,at-2)))==line.substr(at+8),"fallback reference hash");continue;}
        const auto c=split(line);expect(c.size()==14,"fallback fixture fields");++count;environment();Remote remote(c);setenv("FALLBACK_SERVER",c[2]=="1"?remote.url.c_str():"",1);setenv("FALLBACK_MODEL",unhex(c[8]).c_str(),1);
        auto backend=std::make_unique<Backend>();auto *local=backend.get();ezl_env_config config{};ezl_load_config(&config);config.request_timeout_ms=5000;
        ezlocalai::NativeServer server(config,std::move(backend));
        const auto response=server.handle_request("POST",c[0]=="chat"?"/v1/chat/completions":"/v1/completions","",unhex(c[1]));
        // Without FALLBACK_SERVER the source answers HTTP 200 "Local resources
        // exhausted"; native returns a router-retryable status instead.
        const bool unconfigured=c[2]!="1";
        const int status=unconfigured?503:std::stoi(c[10]);const auto type=unconfigured?std::string("application/json"):unhex(c[11]);
        const auto expected=unconfigured?std::string(R"({"detail":{"type":"inference_error","message":"local inference failed"}})"):unhex(c[12]);
        expect(response.status==status&&response.content_type==type&&response.body==expected,"post-fallback case "+std::to_string(count)+": "+std::to_string(response.status)+" "+response.body+" expected "+std::to_string(status)+" "+expected);
        expect(local->calls==1,"one local attempt before fallback");expect(remote.trace()==unhex(c[13]),"post-fallback trace "+std::to_string(count)+": "+remote.trace()+" expected "+unhex(c[13]));
    }
    expect(count==252,"all post-inference source cases: " + std::to_string(count));std::cout<<count<<" post-inference source API, payload and request-trace cases passed\n";return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
