#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "whisper_transcription_test.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
using namespace transcription_test;
std::string hex(const std::string &value){std::string result;for(unsigned char c:value){result+="0123456789abcdef"[c>>4];result+="0123456789abcdef"[c&15];}return result;}
struct Remote {
    Json wires,trace=Json::array();std::mutex mutex;int fd=-1;std::thread thread;std::atomic<bool> stopped{false};size_t at=0;std::string url;
    explicit Remote(Json wires):wires(std::move(wires)){
        fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);expect(fd>=0,"voice remote socket");sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);expect(!bind(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"voice remote bind");socklen_t size=sizeof(address);getsockname(fd,reinterpret_cast<sockaddr *>(&address),&size);url="http://127.0.0.1:"+std::to_string(ntohs(address.sin_port));expect(!listen(fd,16),"voice remote listen");
        thread=std::thread([this]{while(!stopped){int peer=accept(fd,nullptr,nullptr);if(peer<0)continue;serve(peer);shutdown(peer,SHUT_RDWR);close(peer);}});
    }
    ~Remote(){stopped=true;shutdown(fd,SHUT_RDWR);thread.join();close(fd);}
    Json requests(){std::lock_guard<std::mutex> lock(mutex);return trace;}
    void serve(int peer){
        timeval timeout{3,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));std::string wire;char buffer[4096];size_t end=0,length=0;Json headers=Json::object();
        for(;;){const auto n=recv(peer,buffer,sizeof(buffer),0);if(n<=0)return;wire.append(buffer,n);end=wire.find("\r\n\r\n");if(end==std::string::npos)continue;
            std::istringstream lines(wire.substr(0,end));std::string line;std::getline(lines,line);while(std::getline(lines,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();const auto colon=line.find(':');if(colon==std::string::npos)continue;auto key=line.substr(0,colon);for(auto &c:key)c=std::tolower(static_cast<unsigned char>(c));const auto start=line.find_first_not_of(" \t",colon+1);headers[key]=start==std::string::npos?"":line.substr(start);}
            length=std::stoul(headers.value("content-length","0"));if(wire.size()>=end+4+length)break;
        }
        std::istringstream first(wire);std::string method,path;first>>method>>path;headers["host"]="{remote}";
        {std::lock_guard<std::mutex> lock(mutex);trace.push_back({{"method",method},{"path",path},{"headers",headers},{"hex",hex(wire.substr(end+4,length))}});}
        const auto response=unhex(wires.at(std::min(at++,wires.size()-1)));
        for(size_t sent=0;sent<response.size();){const auto n=send(peer,response.data()+sent,response.size()-sent,MSG_NOSIGNAL);if(n<=0)return;sent+=n;}
    }
};
int main(int argc,char **argv){try{
    expect(argc==3,"source and remote voice fixture");const auto cases=records(argv[2]);expect(hash(read(fs::path(argv[1])/"app.py"))==cases[0].at("source_sha256")&&hash(read(fs::path(argv[1])/"Pipes.py"))==cases[0].at("pipes_sha256"),"unchanged voice endpoints and client");expect(cases.size()-1==cases[0].at("cases"),"all remote voice cases");
    Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);fs::create_directories("voices");std::ofstream("voices/local.wav")<<"local";
    size_t compared=0,failed=0;
    for(size_t i=1;i<cases.size();++i){try{const auto &row=cases[i];Remote remote(row.at("wires"));double now=1000;
        for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
        for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","fixture-text"},{"LAZY_LOAD_VOICE","true"},{"ROUTER_URL",""},{"FALLBACK_SERVER",""},{"TTS_ENABLED","true"}})setenv(key.c_str(),value.c_str(),1);
        auto configure=[&](const Json &env){for(const auto &[key,value]:env.items()){auto text=value.get<std::string>();const auto at=text.find("{remote}");if(at!=std::string::npos)text.replace(at,8,remote.url);setenv(key.c_str(),text.c_str(),1);}};configure(row.at("environment"));
        auto probes=native_resource_probes();probes.now=[&]{return now;};probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 16.;};auto resources=std::make_shared<ResourceManager>(probes);
        ezl_env_config config{};ezl_load_config(&config);config.api_key[0]='\0';NativeServer server(config,std::make_unique<EchoInferenceBackend>("fixture-text"),resources);
        for(size_t j=0;j<row.at("actions").size();++j){const auto &action=row.at("actions")[j],&expected=row.at("results")[j];now=action.at("now");configure(action.value("environment",Json::object()));const auto method=action.value("method","GET");
            const auto label="voice remote "+std::to_string(i)+" "+row.at("name").get<std::string>()+" action "+std::to_string(j);const auto result=server.handle_request(method,method=="GET"?"/v1/audio/voices":"/v1/audio/speech","",method=="GET"?"":"{\"input\":\"Hello.\"}","application/json");
            compare_json(result.status,expected.at("status"),label+" status");compare_json(result.content_type,expected.at("content_type"),label+" type");
            if(result.content_type=="application/json")compare_json(Json::parse(result.body),Json::parse(unhex(expected.at("hex"))),label+" body");else compare_json(result.body,unhex(expected.at("hex")),label+" body");
            compare_json(remote.requests().size(),expected.at("requests"),label+" cached probe count");++compared;
        }
        compare_json(remote.requests(),row.at("trace"),"voice remote "+std::to_string(i)+" wire requests");
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';++failed;}
    }
    expect(failed==0,std::to_string(failed)+" remote voice scenarios failed");
    std::cout<<cases.size()-1<<" source remote voice scenarios / "<<compared<<" calls match responses, caching and wire requests\n";return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
