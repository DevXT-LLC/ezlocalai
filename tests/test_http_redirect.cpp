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
size_t url_cases(const fs::path &path){size_t count=0;std::istringstream input(read(path));
    for(std::string line;std::getline(input,line);){if(line.empty()||line[0]=='#')continue;++count;const auto c=split(line);bool ok=true;ezlocalai::HttpUrl url;std::optional<std::pair<std::string,std::string>> auth;
        try{url=ezlocalai::HttpUrl::parse(unhex(c[1]));if(c[2]!="N")url=url.resolve(unhex(c[2].substr(1)));url.require_http();auth=url.credentials();}catch(const std::exception &){ok=false;}
        const auto label="source URL "+c[0]+" "+unhex(c[1]);expect(ok==(c[3]=="1"),label+" outcome");if(!ok)continue;
        const std::vector<std::string> actual{url.str(),url.target(),url.host_header(),url.host,std::to_string(url.port())};
        for(size_t i=0;i<actual.size();++i){const auto expected=i==4?c[4+i]:unhex(c[4+i]);expect(actual[i]==expected,label+" field "+std::to_string(i)+": "+actual[i]+" != "+expected);}
        expect(auth.has_value()==(c[9]!="N"),label+" credentials presence");if(auth)expect(auth->first+'\n'+auth->second==unhex(c[9].substr(1)),label+" credentials");
        auto origin=url;origin.remove_credentials();origin.path.clear();origin.query.clear();origin.fragment.clear();expect(origin.str()==unhex(c[10]),label+" origin: "+origin.str()+" != "+unhex(c[10]));
    }return count;
}
std::string sha256(const std::string &value){unsigned char digest[32];SHA256(reinterpret_cast<const unsigned char *>(value.data()),value.size(),digest);std::ostringstream out;for(auto c:digest)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return out.str();}
std::vector<std::string> lines(const std::string &value){std::vector<std::string> out;std::istringstream input(value);for(std::string line;std::getline(input,line);)out.push_back(line);return out;}
std::string replace(std::string value,const std::string &from,const std::string &to){size_t pos=0;while((pos=value.find(from,pos))!=std::string::npos){value.replace(pos,from.size(),to);pos+=to.size();}return value;}
struct Step{int delay,stall;std::string wire;};
struct Scenario{
    std::mutex mutex;std::atomic<bool> probe_cookie{false};std::vector<Step> steps;size_t index=0;std::string path,first_body,trace;
    std::map<std::string,std::string> urls;
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
        Step step;{
            std::lock_guard<std::mutex> lock(scenario_.mutex);
            const bool target=(scenario_.index>0&&scenario_.index<scenario_.steps.size())||path==scenario_.path||(path!="/v1/resources"&&path!="/v1/models");
            if(target){
                const auto body=request.substr(end+4,length);if(!body.empty()&&scenario_.first_body.empty())scenario_.first_body=body;
                auto type=headers["content-type"];if(type.find("multipart/form-data;")==0)type="multipart/form-data";
                if(!scenario_.trace.empty())scenario_.trace+='\n';scenario_.trace+=tag_+'\t'+method+'\t'+path+'\t'+headers["authorization"]+'\t'+type+'\t'+headers["cookie"]+'\t'+(body.empty()?"empty":body==scenario_.first_body?"payload":"changed");
                if(scenario_.index<scenario_.steps.size())step=scenario_.steps[scenario_.index++];else step={0,0,"HTTP/1.1 500 Bad\r\nContent-Length: 0\r\n\r\n"};
                step.wire=scenario_.expand(step.wire);
            }else{if(!headers["cookie"].empty())scenario_.probe_cookie=true;const auto body=path=="/v1/resources"?std::string("{\"free_vram_gb\":100}"):std::string("{\"data\":[]}");step={0,0,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nSet-Cookie: probe=isolated; Path=/\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body};}
        }
        pause(step.delay);send(peer,step.wire);pause(step.stall);
    }
};
size_t api_cases(const fs::path &reference,const fs::path &path,size_t expected_count){size_t count=0;std::istringstream input(read(path));
    const std::map<std::string,std::string> paths{{"health","/v1/resources"},{"models","/v1/models"},{"music","/v1/audio/music"},{"audio","/v1/audio/speech"},{"stt","/v1/audio/transcriptions"}};
    for(std::string line;std::getline(input,line);){
        if(line.empty())continue;if(line[0]=='#'){const auto marker=line.find(" sha256=");if(marker!=std::string::npos)expect(sha256(read(reference/line.substr(2,marker-2)))==line.substr(marker+8),"redirect reference source changed");continue;}
        ++count;const auto c=split(line);const auto &role=c[2];Scenario scenario;scenario.path=paths.at(role);
        for(const auto &encoded:lines(unhex(c[7]))){const auto step=split(encoded);scenario.steps.push_back({std::stoi(step[0]),std::stoi(step[1]),unhex(step[2])});}
        Remote a(scenario,"A",c[3]),b(scenario,"B",c[3]);scenario.urls={{"A",a.url},{"B",b.url},{"AUTHORITY",a.url.substr(7)},{"OTHER",b.url.substr(7)},{"IPA",replace(a.url,"localhost","127.0.0.1")},{"IPB",replace(b.url,"localhost","127.0.0.1")}};
        for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());
        setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("API_KEY","",1);setenv("DEFAULT_MODEL","none",1);setenv("MUSIC_ENABLED","false",1);setenv("FALLBACK_API_KEY",unhex(c[5]).c_str(),1);setenv("FALLBACK_SERVER",scenario.expand(unhex(c[4])).c_str(),1);
        if(!c[6].empty()){setenv("REQUEST_TIMEOUT",c[6].c_str(),1);setenv("ACE_STEP_TIMEOUT",c[6].c_str(),1);}
        auto probes=ezlocalai::native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.free_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 128.0;};probes.available_ram=[]{return 2.5;};probes.now=[]{return 100.0;};
        ezl_env_config config{};ezl_load_config(&config);ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),std::make_shared<ezlocalai::ResourceManager>(probes));
        const auto started=std::chrono::steady_clock::now();
        const auto response=role=="music"?server.handle_request("POST",scenario.path,"",R"({"prompt":"test","lyrics":"test","keyscale":"C major","duration":10})"):
            role=="audio"?server.handle_request("POST",scenario.path,"",R"({"input":"test"})"):
            role=="stt"?server.handle_request("POST",scenario.path,"","--fixture\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF test\r\n--fixture--\r\n","multipart/form-data; boundary=fixture"):
            server.handle_request("GET","/v1/fallback/status","","");
        const bool ok=c[8]=="1";const auto expected=unhex(c[9]);const auto label="source redirect "+c[0]+" "+c[1]+" "+role;
        if(role=="health"){expect(response.status==200&&response.body.find(expected)!=std::string::npos,label+" reason: "+response.body);expect(response.body.find(std::string("\"available\":")+(ok?"true":"false"))!=std::string::npos,label+" availability");}
        else if(role=="models")expect(response.status==200&&response.body.find("\"remote_models\":"+(ok?expected:"[]"))!=std::string::npos,label+" models: "+response.body);
        else if(ok)expect(response.status==200&&response.body==expected,label+" response: "+response.body);
        else if(role=="music")expect(response.body.find("Music generation not available.")!=std::string::npos,label+" rejected remote response: "+response.body);
        else expect(response.status!=200,label+" rejected response");
        std::string trace;{std::lock_guard<std::mutex> lock(scenario.mutex);trace=scenario.trace;}
        for(const auto &url:std::vector<std::pair<std::string,std::string>>{{"A",a.url},{"B",b.url}})trace=replace(trace,url.second,"{"+url.first+"}");
        expect(trace==unhex(c[10]),label+" trace:\n"+trace+"\nexpected:\n"+unhex(c[10]));
        expect(response.headers.empty(),label+" upstream metadata stays private");expect(!scenario.probe_cookie,label+" cookies stay within each client method call");
        if(c[1].find("redirect-body")!=std::string::npos||c[1]=="cookie-before-stalled-body")expect(std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<.4,label+" unread redirect body");
        if(!c[6].empty()){const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();expect(elapsed>=.16&&elapsed<.4,label+" cumulative deadline");}
    }
    expect(count==expected_count,"complete real HTTP redirect cases");return count;
}

}
int main(int argc,char **argv){try{expect(argc==3,"reference root and fixture directory required");const auto count=url_cases(fs::path(argv[2])/"http_url_cases.tsv");expect(count==1102,"complete source URL cases");const auto api=api_cases(argv[1],fs::path(argv[2])/"http_redirect_cases.tsv",279)+api_cases(argv[1],fs::path(argv[2])/"http_cookie_api_cases.tsv",217);std::cout<<"Verified "<<count<<" source URL cases and "<<api<<" real HTTP redirect cases\n";return 0;}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
