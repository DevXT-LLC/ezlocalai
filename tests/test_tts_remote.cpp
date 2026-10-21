#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "whisper_transcription_test.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <future>
using namespace transcription_test;
std::string bytes_hex(const std::string &s){std::ostringstream out;for(unsigned char c:s)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return out.str();}
struct Remote {
    Json row,trace=Json::array();std::mutex mutex;std::condition_variable progress;int fd=-1;std::thread thread;std::atomic<bool> stopped{false},disconnected{false};size_t at=0;std::string url;
    explicit Remote(Json row):row(std::move(row)) {
        fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);expect(fd>=0,"remote socket");sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);expect(!bind(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"remote bind");socklen_t size=sizeof(address);getsockname(fd,reinterpret_cast<sockaddr *>(&address),&size);url="http://127.0.0.1:"+std::to_string(ntohs(address.sin_port));expect(!listen(fd,16),"remote listen");
        thread=std::thread([this]{while(!stopped){int peer=accept(fd,nullptr,nullptr);if(peer<0)continue;serve(peer);shutdown(peer,SHUT_RDWR);close(peer);}});
    }
    ~Remote(){stopped=true;shutdown(fd,SHUT_RDWR);thread.join();close(fd);}
    Json requests(){std::lock_guard<std::mutex> lock(mutex);return trace;}
    bool wait_post(){std::unique_lock<std::mutex> lock(mutex);return progress.wait_for(lock,std::chrono::seconds(2),[&]{return trace.size()>=2;});}
    bool wait_disconnect(){std::unique_lock<std::mutex> lock(mutex);return progress.wait_for(lock,std::chrono::seconds(2),[&]{return disconnected.load();});}
    static bool send(int peer,const std::string &wire){for(size_t at=0;at<wire.size();){auto n=::send(peer,wire.data()+at,wire.size()-at,MSG_NOSIGNAL);if(n<=0)return false;at+=n;}return true;}
    void serve(int peer){
        timeval timeout{3,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));std::string wire;char buffer[4096];size_t end=0,length=0;Json headers=Json::object();
        for(;;){const auto n=recv(peer,buffer,sizeof(buffer),0);if(n<=0)return;wire.append(buffer,n);end=wire.find("\r\n\r\n");if(end==std::string::npos)continue;
            std::istringstream lines(wire.substr(0,end));std::string line;std::getline(lines,line);while(std::getline(lines,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();const auto colon=line.find(':');if(colon==std::string::npos)continue;auto key=line.substr(0,colon);for(auto &c:key)c=std::tolower(static_cast<unsigned char>(c));const auto start=line.find_first_not_of(" \t",colon+1);headers[key]=start==std::string::npos?"":line.substr(start);}
            length=std::stoul(headers.value("content-length","0"));if(wire.size()>=end+4+length)break;
        }
        std::istringstream first(wire);std::string method,path;first>>method>>path;headers["host"]="{remote}";
        {std::lock_guard<std::mutex> lock(mutex);trace.push_back({{"method",method},{"path",path},{"headers",headers},{"hex",bytes_hex(wire.substr(end+4,length))}});}
        progress.notify_all();
        if(path=="/v1/audio/voices"||path=="/v1/router/health"){send(peer,unhex(row.at("probe_hex")));return;}
        if(row.contains("block")) {
            if(row.at("block")=="body")send(peer,"HTTP/1.1 200 OK\r\nContent-Length: 999\r\n\r\nfirst");
            const auto count=recv(peer,buffer,sizeof(buffer),0);
            {std::lock_guard<std::mutex> lock(mutex);disconnected=count==0;}
            progress.notify_all();return;
        }
        const auto &steps=row.at("steps");const auto sequence=steps.at(std::min(at++,steps.size()-1));
        for(const auto &part:sequence){std::this_thread::sleep_for(std::chrono::milliseconds(part.at("delay").get<int>()));if(!send(peer,unhex(part.at("hex"))))return;}
    }
};
struct Audio:CttsByteStream {
    Json &calls;size_t at=0;explicit Audio(Json &calls):calls(calls){}
    std::optional<std::string> next()override{calls.push_back({{"op","next"}});const std::vector<std::string> packets={"00000000","410042","00000000"};if(at==packets.size())return {};return unhex(packets[at++]);}
    void close()override{calls.push_back({{"op","close"}});}void request_cancel()override{}
};
int main(int argc,char **argv){try{
    expect(argc==3,"source and remote fixture paths");const auto cases=records(argv[2]);expect(hash(read(fs::path(argv[1])/"app.py"))==cases[0].at("source_sha256"),"unchanged speech endpoints");expect(hash(read(fs::path(argv[1])/"Pipes.py"))==cases[0].at("pipes_sha256"),"unchanged remote clients");expect(cases.size()-1==cases[0].at("cases"),"all remote cases");
    Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);
    size_t compared=0;
    for(size_t index=1;index<cases.size();++index)for(bool deferred:{false,true}){
        const auto &row=cases[index];Remote remote(row);Json calls=Json::array();std::vector<std::function<void()>> cleanup;
        for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
        for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","fixture-text"},{"TTS_UNLOAD_LLM_DURING_GENERATION","false"},{"LAZY_LOAD_VOICE","true"},{"TTS_N_PARALLEL","1"},{"FALLBACK_SERVER",""},{"ROUTER_MODE","false"}})setenv(key.c_str(),value.c_str(),1);
        const auto env=row.at("config").at("environment");for(const auto &[key,value]:env.items()){const auto text=value=="{remote}"?remote.url:value.get<std::string>();setenv(key.c_str(),text.c_str(),1);}
        // The oracle controls detect_local_capabilities independently. "0"
        // disables the advertised local TTS capability without the endpoint's
        // exact "false" gate, reproducing that router-only deployment here.
        if(row.at("provider")=="router"&&env.at("TTS_ENABLED")!="false")setenv("TTS_ENABLED","0",1);
        auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 16.;};auto resources=std::make_shared<ResourceManager>(probes);
        TtsPoolFactory factory=[&](auto manager){auto ops=tts_pool_resources({},manager);ops.can_load=[]{return ModelAdmission{true,"cpu","fixture"};};ops.schedule_cleanup=[&](auto work){cleanup.push_back(std::move(work));};ops.load=[&](const auto &){auto model=std::make_shared<TtsPoolModel>();model->device=[]{return "cpu";};model->close=[]{};model->generate=[&](const auto &request){calls.push_back({{"op","generate"},{"request",Json::parse(request)}});return "UklGRgAB";};model->stream=[&](const auto &request){calls.push_back({{"op","stream"},{"request",Json::parse(request)}});return std::make_unique<Audio>(calls);};return model;};return ops;};
        ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::make_unique<EchoInferenceBackend>("fixture-text"),resources,{},factory,{},true);
        auto result=server.handle_request("POST",row.at("path"),"",row.at("body"),"application/json","","",{},deferred);bool aborted=result.transport_error;
        const auto label="remote "+std::to_string(index)+" "+row.at("provider").get<std::string>()+" "+row.at("case").get<std::string>()+(deferred?" deferred":" buffered");
        if(result.stream_writer){
            for(const auto &request:remote.requests())expect(request.at("path")=="/v1/audio/voices"||request.at("path")=="/v1/router/health",label+" no upstream generation before response headers");
            int sockets[2];expect(!socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets),"writer socket pair");result.stream_writer(sockets[0]);shutdown(sockets[0],SHUT_WR);std::string wire;char buffer[4096];for(ssize_t n;(n=read(sockets[1],buffer,sizeof(buffer)))>0;)wire.append(buffer,n);close(sockets[0]);close(sockets[1]);size_t at=0;result.body.clear();aborted=true;while(at<wire.size()){const auto end=wire.find("\r\n",at);expect(end!=std::string::npos,"chunk length");const auto size=std::stoul(wire.substr(at,end-at),nullptr,16);at=end+2;if(!size){aborted=false;break;}expect(at+size+2<=wire.size(),"complete chunk");result.body+=wire.substr(at,size);at+=size+2;}
        }
        Json headers={{"content-type",result.content_type}};for(auto [key,value]:result.headers){for(auto &c:key)c=std::tolower(static_cast<unsigned char>(c));headers[key]=value;}if(!result.chunked)headers["content-length"]=std::to_string(result.body.size());
        compare_json(result.status,row.at("status"),label+" status");compare_json(headers,row.at("headers"),label+" headers");compare_json(bytes_hex(result.body),row.at("hex"),label+" body");compare_json(aborted,row.at("aborted"),label+" completion");compare_json(calls,row.at("calls"),label+" local operations");compare_json(remote.requests(),row.at("trace"),label+" upstream wire requests");
        expect(server.tts_permits()->available==1&&server.tts_pool_snapshot()->leases.empty(),label+" no leaked admission");for(auto &work:cleanup)work();++compared;
    }
    size_t cancellations=0;
    for(const auto *provider:{"voice","router"})for(const auto *block:{"headers","body"})for(bool destroy:{false,true}) {
        auto row=cases[1];row["block"]=block;Remote remote(row);
        setenv("TTS_ENABLED","0",1);setenv("VOICE_SERVER",std::string(provider)=="voice"?remote.url.c_str():"",1);setenv("ROUTER_URL",std::string(provider)=="router"?remote.url.c_str():"",1);
        ezl_env_config config{};ezl_load_config(&config);auto server=std::make_unique<NativeServer>(config,std::make_unique<EchoInferenceBackend>("fixture-text"));
        auto result=server->handle_request("POST","/v1/audio/speech/stream","","{\"input\":\"blocked\"}","application/json","","",{},true);
        expect(result.status==200&&bool(result.stream_writer),"deferred remote stream selected");
        expect(remote.requests().size()==1,"only health probe before response headers");
        int sockets[2];expect(!socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets),"blocked remote pair");
        auto writer=std::async(std::launch::async,[&]{result.stream_writer(sockets[0]);});
        expect(remote.wait_post(),"upstream POST started");
        std::string received;
        if(std::string(block)=="body") {
            timeval timeout{2,0};setsockopt(sockets[1],SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
            char buffer[128];
            while(received.find("5\r\nfirst\r\n")==std::string::npos) {
                const auto count=recv(sockets[1],buffer,sizeof(buffer),0);
                expect(count>0,"first complete audio chunk received before body-phase cancellation");received.append(buffer,count);
            }
        }
        if(destroy)server.reset();else{close(sockets[1]);sockets[1]=-1;}
        expect(writer.wait_for(std::chrono::seconds(2))==std::future_status::ready,"blocked remote read interrupted");writer.get();
        expect(remote.wait_disconnect(),"upstream socket closed after cancellation");
        if(destroy) {
            shutdown(sockets[0],SHUT_WR);char buffer[128];
            for(ssize_t count;(count=recv(sockets[1],buffer,sizeof(buffer),0))>0;)received.append(buffer,count);
            expect(received.find("0\r\n\r\n")==std::string::npos,"cancellation does not complete even a voice-server transfer");
        }
        close(sockets[0]);if(sockets[1]>=0)close(sockets[1]);++cancellations;
    }
    std::cout<<cases.size()-1<<" unchanged-source remote HTTP scenarios; "<<compared<<" native comparisons and "<<cancellations<<" blocked-stream cancellations\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
