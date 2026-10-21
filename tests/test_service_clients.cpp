#include "ezlocalai/configuration.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace {
using namespace ezlocalai; using Json=nlohmann::json;
void check(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
void reset(){
 for(const auto &entry:environment_defaults)unsetenv(std::string(entry.first).c_str());
 for(const auto &[k,v]:std::map<std::string,std::string>{{"DEFAULT_MODEL","fixture-text"},{"EZLOCALAI_API_KEY",""},{"ROUTER_URL",""},{"TEXT_SERVER",""},{"IMAGE_SERVER",""},{"VOICE_SERVER",""},{"EMBEDDING_SERVER",""},{"TTS_ENABLED","false"},{"STT_ENABLED","false"},{"EMBEDDING_ENABLED","false"},{"IMAGE_ENABLED","false"},{"VIDEO_ENABLED","false"},{"MUSIC_ENABLED","false"},{"LAZY_LOAD_VOICE","true"}})setenv(k.c_str(),v.c_str(),1);
}
std::shared_ptr<ResourceManager> manager(){
 auto p=native_resource_probes();p.gpu_count=[]{return 0;};p.total_per_gpu=p.free_per_gpu=[]{return std::vector<double>{};};p.free_vram=[](auto){return 0.;};p.allocator_stats=[]{return std::vector<CudaAllocatorStats>{};};p.system_ram=p.available_ram=[]{return 16.;};return std::make_shared<ResourceManager>(p);
}
struct Backend:InferenceBackend{
 std::string chat(const std::string &)override{return R"({"local":true})";}
 std::string completion(const std::string &)override{return R"({"local":true})";}
 std::string embedding(const std::string &)override{return R"({"local":true,"data":[]})";}
};
struct Capture{
 Json reply;
 int fd=-1,port=0;std::atomic<bool> stop{false};std::mutex mutex;std::vector<std::string> wires;std::string error;std::thread thread;
 explicit Capture(Json response=Json::object()):reply(std::move(response)){
  fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);check(fd>=0,"socket");sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
  check(bind(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))==0&&listen(fd,8)==0,"listen");socklen_t size=sizeof(addr);getsockname(fd,reinterpret_cast<sockaddr*>(&addr),&size);port=ntohs(addr.sin_port);
  thread=std::thread([this]{try{while(!stop){pollfd ready{fd,POLLIN,0};if(poll(&ready,1,20)<=0)continue;int peer=accept(fd,nullptr,nullptr);check(peer>=0,"accept");struct Close{int fd;~Close(){close(fd);}}close_peer{peer};timeval timeout{3,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
   std::string wire;char buf[4096];for(;;){auto n=recv(peer,buf,sizeof(buf),0);check(n>0,"recv");wire.append(buf,n);auto end=wire.find("\r\n\r\n");if(end==std::string::npos)continue;auto pos=wire.find("Content-Length: ");auto length=pos==std::string::npos?0:std::stoul(wire.substr(pos+16));if(wire.size()>=end+4+length)break;}
   {std::lock_guard<std::mutex>lock(mutex);wires.push_back(wire);}
   std::string body=R"({"remote":true,"data":[]})",type="application/json",encoding;int status=200;
   if(!reply.empty()&&wire.rfind("POST ",0)==0){body.clear();auto hex=reply.at("response_body_hex").get<std::string>();for(size_t i=0;i<hex.size();i+=2)body+=static_cast<char>(std::stoi(hex.substr(i,2),nullptr,16));type=reply.at("response_type");status=reply.at("response_status");encoding=reply.at("response_encoding");}
   auto response="HTTP/1.1 "+std::to_string(status)+" Fixture\r\nContent-Type: "+type+"\r\nContent-Length: "+std::to_string(body.size())+"\r\n"+(encoding.empty()?"":"Content-Encoding: "+encoding+"\r\n")+"Connection: close\r\n\r\n"+body;
   check(send(peer,response.data(),response.size(),MSG_NOSIGNAL)==static_cast<ssize_t>(response.size()),"send");
  }}catch(const std::exception&e){std::lock_guard<std::mutex>lock(mutex);error=e.what();}});
 }
 ~Capture(){stop=true;if(thread.joinable())thread.join();close(fd);}
 std::string url()const{return "http://127.0.0.1:"+std::to_string(port)+"/Base///";}
 std::vector<std::string> received(){std::lock_guard<std::mutex>lock(mutex);check(error.empty(),error);return wires;}
};
void settings(const Json &rows){for(const auto &row:rows){reset();for(const auto &[k,v]:row.at("environment").items())setenv(k.c_str(),v.get<std::string>().c_str(),1);
 auto c=capture_service_client_settings(row.at("service"),row.at("override_url"),row.at("override_key"));Json headers{{"Content-Type","application/json"}};if(!c.api_key.empty()&&c.api_key!="none")headers["Authorization"]="Bearer "+c.api_key;
 Json actual{{"raw",c.raw},{"url",c.url},{"api_key",c.api_key},{"dedicated",c.dedicated},{"configured",!c.url.empty()&&!c.dedicated},{"headers",headers}};
 check(actual==row.at("expected"),"client settings "+row.at("service").get<std::string>()+" "+actual.dump());check(Json(local_capabilities())==row.at("capabilities"),"source capability distinction");
}}
void lifetime(const std::string &kind){reset();Capture first,second;const auto key=kind+"_API_KEY";setenv(kind.c_str(),first.url().c_str(),1);setenv(key.c_str(),"first-key",1);setenv("EMBEDDING_ENABLED","true",1);
 ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::make_unique<Backend>(),manager());
 const auto path=kind=="IMAGE_SERVER"?"/v1/images/generations":kind=="TEXT_SERVER"?"/v1/chat/completions":"/v1/embeddings";
 const auto payload=kind=="IMAGE_SERVER"?R"({"prompt":"fixture"})":kind=="TEXT_SERVER"?R"({"model":"fixture-text","messages":[]})":R"({"input":"fixture"})";
 auto send=[&](NativeServer &target){auto response=target.handle_request("POST",path,"",payload);check(response.status==200&&Json::parse(response.body).value("remote",false),"forwarded response: "+response.body);};
 send(server);setenv(kind.c_str(),second.url().c_str(),1);setenv(key.c_str(),"second-key",1);send(server);
 auto old=first.received();check(old.size()==3,"one cached availability probe and two forwarded requests");for(const auto &wire:old){check(wire.find("Authorization: Bearer first-key\r\n")!=std::string::npos,"captured authentication");check(wire.find(" /Base/v1/")!=std::string::npos,"preserved URL path case and slash normalization");}
 check(second.received().empty(),"existing client does not adopt new URL");NativeServer fresh(config,std::make_unique<Backend>(),manager());send(fresh);auto newer=second.received();check(newer.size()==2,"new application uses changed URL");for(const auto &wire:newer)check(wire.find("Authorization: Bearer second-key\r\n")!=std::string::npos,"new authentication");
 if(kind!="TEXT_SERVER"){
  setenv(kind.c_str(),"",1);auto resources=Json::parse(server.handle_request("GET","/v1/resources","","").body);const auto &caps=resources.at("slots").at("cap_slots");check(!caps.contains(kind=="IMAGE_SERVER"?"image":"embedding"),"resource slots retain captured delegation");
 }
}
void local_lifetime(){
 reset();Capture remote;setenv("IMAGE_ENABLED","true",1);setenv("IMG_MODEL","fixture-image",1);setenv("EMBEDDING_ENABLED","true",1);
 ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::make_unique<Backend>(),manager());
 auto chat=[&]{auto response=server.handle_request("POST","/v1/chat/completions","",R"({"model":"fixture-text","messages":[]})");check(response.status==200&&Json::parse(response.body).value("local",false),"captured local text client");};
 chat();server.handle_request("GET","/v1/resources","","");
 for(auto key:{"IMAGE_SERVER","TEXT_SERVER","EMBEDDING_SERVER"})setenv(key,remote.url().c_str(),1);
 chat();auto embedding=server.handle_request("POST","/v1/embeddings","",R"({"input":"fixture"})");check(embedding.status==200&&Json::parse(embedding.body).value("local",false),"captured local embedding client");
 auto resource=Json::parse(server.handle_request("GET","/v1/resources","","").body);check(resource["slots"]["cap_slots"].contains("image")&&resource["slots"]["cap_slots"].contains("embedding"),"local slots remain local after URL edit");check(remote.received().empty(),"no forwarding from captured local clients");
 reset();int calls=0;auto delegated=[&]{++calls;return true;};auto secondary=[]{return std::optional<int>{};};
 setenv("IMAGE_UNLOAD_LLM_DURING_GENERATION","false",1);setenv("VIDEO_UNLOAD_LLM_DURING_GENERATION","false",1);
 check(!image_should_unload_llm(true,false,secondary,[]{return 0.;},delegated)&&!video_should_unload_llm(true,secondary,delegated)&&calls==0,"forced policy skips client initialization");
 setenv("IMAGE_UNLOAD_LLM_DURING_GENERATION","auto",1);setenv("VIDEO_UNLOAD_LLM_DURING_GENERATION","auto",1);
 check(!image_should_unload_llm(true,false,secondary,[]{return 0.;},delegated)&&!video_should_unload_llm(true,secondary,delegated)&&calls==2,"automatic policies use captured forwarding predicate");
}
void response_cases(const Json &rows){for(const auto &row:rows){reset();Capture remote(row);auto url=remote.url();url.replace(0,4,"HTTP");auto service=row.at("service").get<std::string>();setenv(service.c_str(),url.c_str(),1);setenv((service+"_API_KEY").c_str(),"none",1);setenv("EMBEDDING_ENABLED","true",1);
 ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::make_unique<Backend>(),manager());auto response=server.handle_request("POST",row.at("path"),"",row.at("request").dump());auto expected=row.at("expected");
 if(expected.at("action")=="response")check(response.status==expected.at("status")&&response.body==expected.at("body"),row.at("name").get<std::string>()+": "+std::to_string(response.status)+" "+response.body);
 else if(service!="IMAGE_SERVER")check(response.status==200&&Json::parse(response.body).value("local",false),"source client fallback: "+row.at("name").get<std::string>());
 else check(response.body.find("remote")==std::string::npos,"image/video client fallback");
 auto wires=remote.received();check(wires.size()==2,"uppercase URL scheme reaches health and inference: "+row.at("name").get<std::string>());for(const auto &wire:wires)check(wire.find("Authorization:")==std::string::npos,"none omits authentication on wire");
}}
void unreachable_image(){reset();setenv("IMAGE_SERVER","none",1);setenv("IMAGE_ENABLED","true",1);setenv("VIDEO_ENABLED","true",1);setenv("IMG_MODEL","fixture-image",1);setenv("VIDEO_MODEL","wan2.2",1);setenv("IMAGE_UNLOAD_LLM_DURING_GENERATION","false",1);
 ezl_env_config config{};ezl_load_config(&config);auto resources=manager();NativeServer server(config,std::make_unique<Backend>(),resources);
 auto result=server.handle_request("POST","/v1/images/generations","",R"({"prompt":"fixture"})");check(result.status==200&&Json::parse(result.body).at("data").at(0).at("url")=="","configured unreachable image client does not load local image model");
 setenv("IMAGE_SERVER","",1);result=server.handle_request("POST","/v1/images/edits","",R"({"prompt":"fixture","image":"fixture-image"})");check(result.status==503,"captured delegation suppresses local image edit after environment mutation");check(!resources->is_model_loaded(ModelType::Image),"no local image residency");
 result=server.handle_request("POST","/v1/videos/generations","",R"({"prompt":"fixture"})");check(result.status==500,"captured delegation suppresses local video after environment mutation");
 auto status=Json::parse(server.handle_request("GET","/v1/resources","","").body);check(!status["slots"]["cap_slots"].contains("image")&&!status["slots"]["cap_slots"].contains("video"),"captured image/video slots");
}
}
int main(int argc,char**argv){try{check(argc==3,"fixtures required");std::ifstream input(argv[1]),responses_input(argv[2]);Json fixture,responses;input>>fixture;responses_input>>responses;setenv("CUDA_VISIBLE_DEVICES","",1);char pattern[]="/tmp/ezlocalai-service-clients-XXXXXX";check(mkdtemp(pattern),"temp directory");auto old=std::filesystem::current_path();std::filesystem::current_path(pattern);struct Cleanup{std::filesystem::path old,path;~Cleanup(){std::filesystem::current_path(old);std::error_code e;std::filesystem::remove_all(path,e);}}cleanup{old,pattern};settings(fixture.at("cases"));for(auto kind:{"IMAGE_SERVER","TEXT_SERVER","EMBEDDING_SERVER"})lifetime(kind);local_lifetime();unreachable_image();response_cases(responses.at("cases"));std::cout<<fixture.at("cases").size()<<" source settings, "<<responses.at("cases").size()<<" source HTTP responses and native forwarding lifetime checks passed\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
