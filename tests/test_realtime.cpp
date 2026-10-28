#include "ezlocalai/native_server.hpp"
#include "ezlocalai/tunnel_websocket.hpp"
#include "fixture_ports.hpp"
#include <nlohmann/json.hpp>
#include "ezlocalai/realtime.hpp"
#include "ezlocalai/omni_model.hpp"
#include "ezlocalai/llm_cache.hpp"
#include "ezlocalai/llm_catalog.hpp"
#include "ezlocalai/llm_request.hpp"
#include "ezlocalai/model_download.hpp"
#include "ezlocalai/http_socket.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <future>
#include <iostream>
#include <set>
#include <poll.h>
#include <thread>
#include <unistd.h>
using namespace ezlocalai;
namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
struct Fd { int fd=-1; ~Fd(){if(fd>=0){shutdown(fd,SHUT_RDWR);close(fd);}} };
std::string receive(int fd,size_t size) {
    std::string out;char buffer[4096];
    while(out.size()<size){auto n=recv(fd,buffer,std::min(sizeof(buffer),size-out.size()),0);require(n>0,"short fixture read");out.append(buffer,size_t(n));}
    return out;
}
void profiles() {
    require(realtime_audio_model("LiquidAI/LFM3-Omni-2B-GGUF"),"official omni family");
    require(realtime_audio_model("/models/LFM3-Omni-2B-Q8_0.gguf"),"local omni family");
    require(!realtime_audio_model("other/LFM3-Omni-2Bigger")&&!realtime_audio_model("Qwen-Omni"),"no substring capability guesses");
    auto env=[](const std::string &,const std::string &fallback){return fallback;};
    require(configured_llm_kv_cache_type("3090",env,"LiquidAI/LFM3-Omni-2B-GGUF")=="f16","full precision voice cache default");
    require(configured_llm_kv_cache_type("3090",env,"Qwen3.8-27B")=="q4_0","other defaults preserved");
    auto explicit_cache=[](const std::string &key,const std::string &fallback){return key=="KV_CACHE_TYPE"?"q8_0":fallback;};
    require(configured_llm_kv_cache_type("3090",explicit_cache,"LFM3-Omni-2B")=="q8_0","explicit cache override retained");
    LlmCatalogSettings settings;settings.models="LiquidAI/LFM3-Omni-2B-GGUF";settings.contexts="230000";settings.parallels="0";
    auto c=parse_llm_catalog(settings);require(c.configs.begin()->second.context==128000&&c.configs.begin()->second.parallel==1,"published context cap and voice auto parallel");
    settings.models="unsloth/Qwen3.8-27B-GGUF";settings.contexts="262144";settings.parallels="1";
    require(parse_llm_catalog(settings).configs.begin()->second.context==262144,"Qwen 262144 context remains unchanged");
    settings.models="LiquidAI/LFM3-Omni-2B-GGUF";
    const auto defaults=llm_model_request_settings(R"({"messages":[]})",settings.models);
    require(defaults.find("\"temperature\":0.0")!=std::string::npos&&defaults.find("\"enable_thinking\":false")!=std::string::npos,"nonreasoning greedy text profile");
    const auto explicit_request=llm_model_request_settings(R"({"messages":[],"temperature":0.5})",settings.models);
    require(explicit_request.find("\"temperature\":0.5")!=std::string::npos,"explicit sampling preserved");
}
void acquisition() {
    const std::string weight="LFM3-Omni-2B-F16.gguf",projector="mmproj-LFM3-Omni-2B-F16.gguf";
    std::set<std::string> downloaded;int listings=0;
    ModelDownloadHooks io;io.mkdir=[](const auto &){};io.list_local=[](const auto &){return std::vector<std::string>{};};io.exists=[&](const auto &s){return downloaded.count(s);};
    io.list_remote=[&](const auto &repo){require(repo=="LiquidAI/LFM3-Omni-2B-GGUF","official repository");++listings;return std::vector<std::string>{"LFM3-Omni-2B-Q4_K_M.gguf",weight,projector};};
    io.download=[&](const auto &,const auto &file,const auto &dir){downloaded.insert(dir+"/"+file);};
    auto f=download_gguf_model("LFM3-Omni-2B","/models",std::string("UD-Q3_K_XL"),false,io);
    require(f.model=="/models/LFM3-Omni-2B/"+weight&&f.projector=="/models/LFM3-Omni-2B/"+projector,"backbone and model-qualified codec selected");
    require(listings==1&&downloaded.size()==2,"single shared catalogue");
    io.list_local=[&](const auto &){return std::vector<std::string>{weight,projector};};io.list_remote=[](const auto &)->std::vector<std::string>{throw std::runtime_error("offline");};
    auto cached=download_gguf_model("LiquidAI/LFM3-Omni-2B-GGUF","/models",std::string("F16"),false,io);require(cached.model==f.model&&cached.projector==f.projector,"offline asset reuse");
    io.list_local=[](const auto &){return std::vector<std::string>{};};io.list_remote=[&](const auto &){return std::vector<std::string>{weight};};
    bool rejected=false;try{download_gguf_model("LFM3-Omni-2B","/models",{},false,io);}catch(const std::exception &){rejected=true;}require(rejected,"missing audio projector cannot silently degrade");
}
std::string frame(unsigned char opcode,const std::string &payload,bool mask,bool fin=true) {
    std::string out(1,char((fin?128:0)|opcode));auto size=payload.size();
    if(size<126)out+=char((mask?128:0)|size);
    else if(size<=65535){out+=char((mask?128:0)|126);out+=char(size>>8);out+=char(size);}
    else{out+=char((mask?128:0)|127);for(int i=56;i>=0;i-=8)out+=char(uint64_t(size)>>i);}
    const unsigned char key[]={17,34,51,68};if(mask)out.append(reinterpret_cast<const char *>(key),4);
    for(size_t i=0;i<size;++i)out+=char(static_cast<unsigned char>(payload[i])^(mask?key[i%4]:0));return out;
}
void duplex() {
    Fd listener{socket(AF_INET,SOCK_STREAM,0)};require(listener.fd>=0,"listener");
    sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    require(!bind(listener.fd,reinterpret_cast<sockaddr *>(&a),sizeof(a))&&!listen(listener.fd,1),"listen");socklen_t len=sizeof(a);getsockname(listener.fd,reinterpret_cast<sockaddr *>(&a),&len);
    std::string bytes(256*1024,'\0');for(size_t i=0;i<bytes.size();++i)bytes[i]=char(i%251);
    const auto input=frame(1,R"({"type":"response.cancel"})",true)+frame(2,bytes.substr(0,180000),true,false)+frame(0,bytes.substr(180000),true)+frame(9,"keepalive",true)+frame(8,std::string("\x03\xe8",2),true);
    const auto output=frame(1,R"({"type":"session.created"})",false)+frame(2,bytes,false)+frame(8,std::string("\x03\xe8",2),false);
    auto server=std::async(std::launch::async,[&]{
        pollfd waiting{listener.fd,POLLIN,0};require(poll(&waiting,1,10000)>0,"bounded accept");
        Fd peer{accept(listener.fd,nullptr,nullptr)};require(peer.fd>=0,"accept");timeval timeout{10,0};setsockopt(peer.fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(peer.fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));int small=4096;setsockopt(peer.fd,SOL_SOCKET,SO_SNDBUF,&small,sizeof(small));
        std::string head;while(head.find("\r\n\r\n")==std::string::npos)head+=receive(peer.fd,1);
        require(head.find("GET /v1/realtime?intent=transcription HTTP/1.1")!=std::string::npos&&head.find("worker_id=")==std::string::npos,"ordinary endpoint query is unchanged");require(head.find("Authorization: Bearer engine-key")!=std::string::npos,"private engine authentication");
        auto k=head.find("Sec-WebSocket-Key: ");require(k!=std::string::npos,"upgrade key");k+=19;auto key=head.substr(k,head.find("\r\n",k)-k);
        require(http_socket::send_all(peer.fd,"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "+tunnel_websocket_accept(key)+"\r\n\r\n"),"upgrade response");
        auto writer=std::async(std::launch::async,[&]{require(http_socket::send_all(peer.fd,output),"backpressured model output");});
        const auto received=receive(peer.fd,input.size());writer.get();require(received==input,"client masks, fragmentation, control and binary bytes preserved");
    });
    std::atomic<bool> stop{false};TunnelWebSocketConnection connection([&]{return stop.load();});std::string initial,error;
    require(connection.connect_endpoint("http://127.0.0.1:"+std::to_string(ntohs(a.sin_port))+"/v1/realtime?intent=transcription","engine-key",&initial,&error),error.c_str());
    int pair[2];require(!socketpair(AF_UNIX,SOCK_STREAM,0,pair),"frontend socketpair");Fd front{pair[0]},client{pair[1]};timeval timeout{10,0};setsockopt(client.fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    auto relay=std::async(std::launch::async,[&]{relay_realtime_websocket(front.fd,connection,"",initial,[&]{return stop.load();},5);});
    struct Stop {std::atomic<bool>&s;~Stop(){s=true;}} cleanup{stop};
    auto reader=std::async(std::launch::async,[&]{return receive(client.fd,output.size());});
    require(http_socket::send_all(client.fd,input),"frontend write");require(reader.get()==output,"upstream bytes preserved through bounded relay");server.get();stop=true;relay.get();
}
struct BackendState { std::atomic<int> active{0}, calls{0}; std::atomic<bool> fail{false}; std::string url; };
struct Backend final : InferenceBackend {
    std::shared_ptr<BackendState> state;
    explicit Backend(std::shared_ptr<BackendState> value):state(std::move(value)){}
    std::string chat(const std::string &) override{return "{}";}
    std::string completion(const std::string &) override{return "{}";}
    std::string embedding(const std::string &) override{return "{}";}
    std::unique_ptr<RealtimeEngineLease> open_realtime(const std::string &model,const std::string &) override {
        ++state->calls;require(model=="LiquidAI/LFM3-Omni-2B-GGUF","correct realtime model selected");
        if(state->fail)throw std::runtime_error("fixture model unavailable");
        auto lease=std::make_unique<RealtimeEngineLease>();lease->url=state->url;lease->api_key="engine-key";
        auto saved=state;lease->release=[saved]{--saved->active;};++state->active;return lease;
    }
};
void gateway() {
    for(const auto &entry:std::vector<std::pair<std::string,std::string>>{
        {"DEFAULT_MODEL","LiquidAI/LFM3-Omni-2B-GGUF"},{"EZLOCALAI_API_KEY","front-key"},{"N_PARALLEL","1"},
        {"ROUTER_URL",""},{"ROUTER_MODE","false"},{"TTS_ENABLED","false"},{"STT_ENABLED","false"},
        {"EMBEDDING_ENABLED","false"},{"IMAGE_ENABLED","false"},{"VIDEO_ENABLED","false"},{"MUSIC_ENABLED","false"}})
        setenv(entry.first.c_str(),entry.second.c_str(),1);
    const auto port=fixture_ports(2);
    Fd listener{socket(AF_INET,SOCK_STREAM,0)};sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);a.sin_port=htons(port);
    require(!bind(listener.fd,reinterpret_cast<sockaddr *>(&a),sizeof(a))&&!listen(listener.fd,1),"gateway backend listener");
    auto backend=std::async(std::launch::async,[&]{
        pollfd waiting{listener.fd,POLLIN,0};require(poll(&waiting,1,10000)>0,"bounded gateway accept");Fd peer{accept(listener.fd,nullptr,nullptr)};
        timeval timeout{10,0};setsockopt(peer.fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        std::string head;while(head.find("\r\n\r\n")==std::string::npos)head+=receive(peer.fd,1);
        auto k=head.find("Sec-WebSocket-Key: ");require(k!=std::string::npos,"gateway engine key");k+=19;const auto key=head.substr(k,head.find("\r\n",k)-k);
        require(head.find("Authorization: Bearer engine-key")!=std::string::npos,"engine secret independent of worker secret");
        require(http_socket::send_all(peer.fd,"HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Accept: "+tunnel_websocket_accept(key)+"\r\n\r\n"+frame(1,R"({"type":"session.created"})",false)),"gateway backend ready");
        const auto header=receive(peer.fd,2);require((static_cast<unsigned char>(header[1])&128)!=0,"client frame remains masked");
        size_t size=static_cast<unsigned char>(header[1])&127;require(size<126,"small control fixture");auto mask=receive(peer.fd,4);auto body=receive(peer.fd,size);for(size_t i=0;i<body.size();++i)body[i]=char(static_cast<unsigned char>(body[i])^static_cast<unsigned char>(mask[i%4]));
        require(body==R"({"type":"response.cancel"})","cancel reaches engine without rewriting");
        require(http_socket::send_all(peer.fd,frame(1,body,false)),"gateway echo");
        // The client closes abruptly; the worker must release its model lease.
        char byte;while(recv(peer.fd,&byte,1,0)>0){}
    });
    auto state=std::make_shared<BackendState>();state->url="http://127.0.0.1:"+std::to_string(port)+"/v1/realtime";
    auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[] { return std::vector<double>{}; };probes.free_per_gpu=[] { return std::vector<double>{}; };probes.free_vram=[](auto){return 0.;};probes.system_ram=probes.available_ram=[]{return 16.;};probes.allocator_stats=[] { return std::vector<CudaAllocatorStats>{}; };
    auto resources=std::make_shared<ResourceManager>(probes);ezl_env_config config{};ezl_load_config(&config);
    NativeServer server(config,std::make_unique<Backend>(state),resources);std::string error;require(server.start("127.0.0.1",port+1,&error),error.c_str());
    const auto url="http://127.0.0.1:"+std::to_string(port+1)+"/v1/realtime?model=LiquidAI%2FLFM3-Omni-2B-GGUF";
    TunnelWebSocketConnection client;std::string buffered;
    require(!client.connect_endpoint(url,"wrong-key",&buffered,&error)&&state->calls==0,"unauthorized call rejected before acquiring model");
    require(client.connect_endpoint(url,"front-key",&buffered,&error),error.c_str());
    TunnelWebSocketReader reader;TunnelWebSocketMessage message;
    const auto read_message=[&]{
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(std::chrono::steady_clock::now()<deadline){if(reader.pop(buffered,message))return message.payload;if(client.readable(10)){char bytes[4096];auto n=client.read_some(bytes,sizeof(bytes));if(n>0)buffered.append(bytes,size_t(n));else if(!n)break;}}
        throw std::runtime_error("gateway event timeout");
    };
    require(read_message().find("session.created")!=std::string::npos&&state->active==1,"call retains backend lease");
    auto snapshot=nlohmann::json::parse(server.handle_request("GET","/v1/resources","Bearer front-key","").body);
    require(snapshot["slots"]["model_slots"]["LiquidAI/LFM3-Omni-2B-GGUF"]["in_flight"]==1,"call shares model slot counters");
    require(client.send_text(R"({"type":"response.cancel"})"),"gateway cancel send");require(read_message()==R"({"type":"response.cancel"})","gateway preserves bidirectional event");client.close_all();
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(std::chrono::steady_clock::now()<deadline){
        snapshot=nlohmann::json::parse(server.handle_request("GET","/v1/resources","Bearer front-key","").body);
        if(!state->active&&snapshot["slots"]["model_slots"]["LiquidAI/LFM3-Omni-2B-GGUF"]["in_flight"]==0)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    require(state->active==0,"disconnect releases backend lease");
    snapshot=nlohmann::json::parse(server.handle_request("GET","/v1/resources","Bearer front-key","").body);
    require(snapshot["slots"]["model_slots"]["LiquidAI/LFM3-Omni-2B-GGUF"]["in_flight"]==0,"disconnect releases model slot");
    state->fail=true;require(!client.connect_endpoint(url,"front-key",&buffered,&error),"model startup failure cannot send a successful upgrade");
    snapshot=nlohmann::json::parse(server.handle_request("GET","/v1/resources","Bearer front-key","").body);
    require(snapshot["slots"]["model_slots"]["LiquidAI/LFM3-Omni-2B-GGUF"]["in_flight"]==0,"failed upgrade releases model admission");
    server.stop();backend.get();
}

}
int main(){try{profiles();acquisition();duplex();gateway();std::cout<<"Realtime profile/acquisition and full-duplex framing/backpressure checks passed\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
