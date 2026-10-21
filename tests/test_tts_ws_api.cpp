#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "whisper_transcription_test.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
using namespace transcription_test;
std::string hex(const std::string &value){std::string out;for(unsigned char c:value){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;}
int port(){int fd=socket(AF_INET,SOCK_STREAM,0);sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);expect(!bind(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"reserve port");socklen_t size=sizeof(address);getsockname(fd,reinterpret_cast<sockaddr *>(&address),&size);close(fd);return ntohs(address.sin_port);}
void send_all(int fd,const std::string &bytes){for(size_t at=0;at<bytes.size();){const auto size=send(fd,bytes.data()+at,bytes.size()-at,MSG_NOSIGNAL);expect(size>0,"client send");at+=size;}}
std::string frame(const std::string &bytes,unsigned char opcode=1,bool final=true){std::string out(1,static_cast<char>((final?128:0)|opcode));if(bytes.size()<126)out+=static_cast<char>(128|bytes.size());else{out+=static_cast<char>(128|126);out+=static_cast<char>(bytes.size()>>8);out+=static_cast<char>(bytes.size());}const std::string mask="abcd";out+=mask;for(size_t at=0;at<bytes.size();++at)out+=static_cast<char>(bytes[at]^mask[at%4]);return out;}
Json frames(const std::string &wire){Json out=Json::array();size_t at=0;while(at<wire.size()){expect(at+2<=wire.size(),"response frame header");const auto opcode=static_cast<unsigned char>(wire[at++])&15;uint64_t size=static_cast<unsigned char>(wire[at++])&127;if(size==126){expect(at+2<=wire.size(),"response length");size=(static_cast<unsigned char>(wire[at])<<8)|static_cast<unsigned char>(wire[at+1]);at+=2;}else if(size==127){expect(at+8<=wire.size(),"response long length");size=0;for(unsigned i=0;i<8;++i)size=(size<<8)|static_cast<unsigned char>(wire[at++]);}expect(size<=wire.size()-at,"complete response frame");out.push_back({{"opcode",opcode},{"hex",hex(wire.substr(at,size))}});at+=size;}return out;}
struct Audio:CttsByteStream {
    Json config;Json &calls;size_t at=0;Audio(Json config,Json &calls):config(std::move(config)),calls(calls){}
    std::optional<std::string> next()override{calls.push_back({{"op","next"}});if(int(at)==config.value("next_error_at",-1))throw TtsError("RuntimeError","stream failure");const auto packets=config.value("packets",Json::array({"c05d000010000100","04000000","41004200","00000000"}));if(at==packets.size())return {};return unhex(packets[at++]);}
    void close()override{calls.push_back({{"op","close"}});if(config.contains("close_error"))throw TtsError("RuntimeError",config.at("close_error"));}void request_cancel()override{}
};
int main(int argc,char **argv){try{
    expect(argc==3,"source and WebSocket API fixture");const auto cases=records(argv[2]);expect(hash(read(fs::path(argv[1])/"ezlocalai/TTSWebSocket.py"))==cases[0].at("source_sha256")&&hash(read(fs::path(argv[1])/"app.py"))==cases[0].at("app_sha256"),"unchanged WebSocket endpoint and session");expect(cases.size()-1==cases[0].at("cases"),"all API cases");
    Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);size_t compared=0;
    for(size_t index=1;index<cases.size();++index)for(bool coalesced:{false,true}){
        const auto &row=cases[index];const auto config=row.at("config");Json calls=Json::array();
        for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
        for(const auto &[key,value]:std::map<std::string,std::string>{{"DEFAULT_MODEL","fixture-text"},{"LAZY_LOAD_VOICE","true"},{"TTS_UNLOAD_LLM_DURING_GENERATION","false"},{"TTS_N_PARALLEL","1"},{"VOICE_SERVER","http://127.0.0.1:1"},{"ROUTER_URL",""},{"FALLBACK_SERVER",""},{"ROUTER_MODE","false"}})setenv(key.c_str(),value.c_str(),1);
        setenv("EZLOCALAI_API_KEY",config.value("api_key","secret").c_str(),1);setenv("TTS_ENABLED",config.value("enabled","true").c_str(),1);
        auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 16.;};auto resources=std::make_shared<ResourceManager>(probes);
        TtsPoolFactory factory=[&](auto manager){auto ops=tts_pool_resources({},manager);ops.can_load=[]{return ModelAdmission{true,"cpu","fixture"};};ops.schedule_cleanup=[](auto work){work();};ops.load=[&](const auto &){calls.push_back({{"op","get"}});if(config.contains("get_error"))throw TtsError("RuntimeError",config.at("get_error"));auto model=std::make_shared<TtsPoolModel>();model->device=[]{return "cpu";};model->close=[&]{calls.push_back({{"op","destroy"}});};model->stream=[&](const auto &request){calls.push_back({{"op","stream"},{"request",Json::parse(request)}});if(config.contains("stream_error"))throw TtsError("RuntimeError",config.at("stream_error"));return std::make_unique<Audio>(config,calls);};return model;};return ops;};
        ezl_env_config environment{};ezl_load_config(&environment);NativeServer server(environment,std::make_unique<EchoInferenceBackend>("fixture-text"),resources,{},factory,{},true);const auto listen_port=port();std::string error;expect(server.start("127.0.0.1",listen_port,&error),error);
        int fd=socket(AF_INET,SOCK_STREAM,0);timeval timeout{3,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);address.sin_port=htons(listen_port);expect(!connect(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"connect WebSocket");
        std::string request="GET /v1/audio/speech/ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n";
        if(config.contains("authorization")&&!config.at("authorization").is_null())request+="Authorization: "+config.at("authorization").get<std::string>()+"\r\n";request+="\r\n";
        std::string messages;for(const auto &item:row.at("messages")){const auto text=item.get<std::string>();if(coalesced){const auto cut=text.size()/2;messages+=frame(text.substr(0,cut),1,false)+frame(text.substr(cut),0,true);}else messages+=frame(text);}
        send_all(fd,request+(coalesced?messages:""));std::string wire;char buffer[4096];size_t end;
        while((end=wire.find("\r\n\r\n"))==std::string::npos){const auto n=recv(fd,buffer,sizeof(buffer),0);expect(n>0,"WebSocket handshake received");wire.append(buffer,n);}
        std::istringstream status_line(wire);std::string protocol;int status;status_line>>protocol>>status;if(status==101&&!coalesced)send_all(fd,messages);wire.erase(0,end+4);
        for(;;){const auto n=recv(fd,buffer,sizeof(buffer),0);if(n==0||(n<0&&errno==ECONNRESET))break;expect(n>0,"WebSocket session completed without read timeout");wire.append(buffer,n);}close(fd);server.stop();
        const auto label="WS API "+std::to_string(index)+(coalesced?" coalesced/fragmented":" normal");compare_json(status,row.at("status"),label+" upgrade status");compare_json(status==101?frames(wire):Json::array(),row.at("frames"),label+" frames");compare_json(calls,row.at("calls"),label+" local model calls");
        expect(server.tts_permits()->available==1&&server.tts_pool_snapshot()->leases.empty(),label+" permits and leases released");expect(resources->get_model_active_count(ModelType::Tts)==0,label+" model usage released");++compared;
    }
    std::cout<<cases.size()-1<<" source WebSocket API scenarios, "<<compared<<" TCP comparisons\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
