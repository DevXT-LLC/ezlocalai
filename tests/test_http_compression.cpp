#include "ezlocalai/http_body.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void expect(bool ok,const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path); return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &s) { std::vector<std::string> out; size_t p=0; for (;;) { const auto n=s.find('\t',p); out.push_back(s.substr(p,n==std::string::npos?n:n-p)); if(n==std::string::npos)return out; p=n+1; } }
std::string unhex(const std::string &s) { std::string out; for(size_t i=0;i<s.size();i+=2)out+=static_cast<char>(std::stoi(s.substr(i,2),nullptr,16));return out; }
std::string hash(const std::string &s) { unsigned char bytes[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()),s.size(),bytes);std::ostringstream out;for(auto b:bytes)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);return out.str(); }
size_t decoder_cases(const fs::path &path) {
    size_t count=0;std::istringstream input(read(path));
    for(std::string line;std::getline(input,line);) {
        if(line.empty()||line[0]=='#')continue;
        ++count;const auto c=split(line);const auto bytes=unhex(c[3]);const auto width=std::stoul(c[2]);bool ok=true;std::string body;
        try {
            ezlocalai::HttpBodyDecoder decoder(c[1],false,{});
            for(size_t p=0;p<bytes.size();p+=width?width:bytes.size())decoder.feed(std::string_view(bytes).substr(p,width?width:bytes.size()));
            body=decoder.finish();
        }catch(const std::exception &){ok=false;}
        expect(ok==(c[4]=="1"),"source content decoder outcome "+c[0]+" "+c[1]);
        if(ok)expect(body.size()==std::stoul(c[5])&&hash(body)==c[6],"source content decoder bytes "+c[0]+" "+c[1]);
    }
    expect(count==1407,"complete compression fixtures");return count;
}
class Remote {
public:
    Remote(std::string role,std::string wire,bool stall):role_(std::move(role)),wire_(std::move(wire)),stall_(stall) {
        listener_=socket(AF_INET,SOCK_STREAM,0);expect(listener_>=0,"compression listener");
        sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        expect(!bind(listener_,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"compression bind");socklen_t size=sizeof(address);getsockname(listener_,reinterpret_cast<sockaddr *>(&address),&size);
        url="http://127.0.0.1:"+std::to_string(ntohs(address.sin_port));expect(!listen(listener_,16),"compression listen");
        thread_=std::thread([&]{run();});
    }
    ~Remote(){stopped_=true;shutdown(listener_,SHUT_RDWR);{std::lock_guard<std::mutex> lock(mutex_);if(peer_>=0)shutdown(peer_,SHUT_RDWR);}thread_.join();close(listener_);}
    std::string url;std::atomic<bool> bad_headers{false};std::atomic<int> target_requests{0};
private:
    int listener_=-1,peer_=-1;std::string role_,wire_;bool stall_;std::atomic<bool> stopped_{false};std::thread thread_;std::mutex mutex_;
    bool send(int fd,const std::string &data){for(size_t p=0;p<data.size();){const auto n=::send(fd,data.data()+p,data.size()-p,MSG_NOSIGNAL);if(n<=0)return false;p+=n;}return true;}
    void serve(int peer){
        timeval timeout{2,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));std::string request;char buffer[4096];
        for(;;){const auto n=recv(peer,buffer,sizeof(buffer),0);if(n<=0)return;request.append(buffer,n);const auto end=request.find("\r\n\r\n");if(end==std::string::npos)continue;
            const auto length=request.find("Content-Length: ");const auto bytes=length==std::string::npos?0:std::stoul(request.substr(length+16));if(request.size()>=end+4+bytes)break;}
        if(request.find("Accept-Encoding: gzip, deflate, br\r\n")==std::string::npos||request.find("Accept: */*\r\n")==std::string::npos)bad_headers=true;
        std::istringstream first(request);std::string method,path;first>>method>>path;
        const bool target=role_=="health"?path=="/v1/resources":role_=="models"?path=="/v1/models":method=="POST";
        if(target){++target_requests;send(peer,wire_);if(stall_)for(int i=0;i<200&&!stopped_;++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));}
        else {const std::string body=path=="/v1/resources"?"{\"free_vram_gb\":100}":"{\"data\":[]}";send(peer,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body);}
    }
    void run(){while(!stopped_){const int peer=accept(listener_,nullptr,nullptr);if(peer<0)continue;{std::lock_guard<std::mutex> lock(mutex_);peer_=peer;}serve(peer);{std::lock_guard<std::mutex> lock(mutex_);peer_=-1;shutdown(peer,SHUT_RDWR);close(peer);}}}
};
size_t api_cases(const fs::path &root,const fs::path &path){
    size_t count=0;std::istringstream input(read(path));
    for(std::string line;std::getline(input,line);){
        if(line.empty())continue;
        if(line[0]=='#'){const auto marker=line.find(" sha256=");if(marker!=std::string::npos)expect(hash(read(root/line.substr(2,marker-2)))==line.substr(marker+8),"compression reference source changed");continue;}
        ++count;const auto c=split(line);Remote remote(c[1],unhex(c[2]),c[3]!="0");
        for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());
        setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("API_KEY","",1);setenv("DEFAULT_MODEL","none",1);setenv("MUSIC_ENABLED","false",1);setenv("FALLBACK_SERVER",remote.url.c_str(),1);
        auto probes=ezlocalai::native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.free_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 128.0;};probes.available_ram=[]{return 2.5;};probes.now=[]{return 100.0;};
        ezl_env_config config{};ezl_load_config(&config);ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),std::make_shared<ezlocalai::ResourceManager>(probes));
        if(c[3]=="2"){setenv("REQUEST_TIMEOUT","0.06",1);setenv("ACE_STEP_TIMEOUT","0.06",1);}
        const auto start=std::chrono::steady_clock::now();
        const auto response=c[1]=="music"?server.handle_request("POST","/v1/audio/music","",R"({"prompt":"test","lyrics":"test","keyscale":"C major","duration":10})"):c[1]=="audio"?server.handle_request("POST","/v1/audio/speech","",R"({"input":"test"})"):server.handle_request("GET","/v1/fallback/status","","");
        const bool ok=c[4]=="1";const auto expected=unhex(c[5]);const auto label="source HTTP compression "+c[0]+" "+c[1];
        if(c[1]=="health"){
            expect(response.status==200&&response.body.find(expected)!=std::string::npos,label+" reason: "+response.body);
            expect(response.body.find(std::string("\"available\":")+(ok?"true":"false"))!=std::string::npos,label+" availability");
        }else if(c[1]=="models")expect(response.status==200&&response.body.find("\"remote_models\":"+(ok?expected:"[]"))!=std::string::npos,label+" models: "+response.body);
        else if(ok)expect(response.status==200&&response.body==expected,label+" forwarded body: "+response.body);
        else if(c[1]=="music")expect(response.body.find("Music generation not available.")!=std::string::npos,label+" rejected remote response");
        else expect(response.status!=200,label+" rejected remote audio");
        if(c[3]=="1")expect(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<.5,label+" detects corruption before stalled body ends");
        if(c[3]=="2"){const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();expect(seconds>=.05&&seconds<.5,label+" deadline survives compressed stream end");}
        expect(!remote.bad_headers&&remote.target_requests==1,label+" request headers/count");
    }
    expect(count==592,"complete real HTTP compression fixtures");return count;
}
}
int main(int argc,char **argv) {
    try {expect(argc==3,"reference checkout and fixture directory required");const auto count=decoder_cases(fs::path(argv[2])/"http_compression_cases.tsv");const auto api=api_cases(argv[1],fs::path(argv[2])/"http_compression_api_cases.tsv");std::cout<<"Verified "<<count<<" source content decoder cases and "<<api<<" real HTTP API cases\n";return 0;}
    catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
